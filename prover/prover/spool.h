// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Class Spool: the spool directory, through which requests reach the proving service (the prover
// program cp_prove with -spool) and responses leave it. CPU code only: it uses no GPU and does not
// interpret a proof's bytes. It also serves a single batch (cp_prove -in, -out): see below.
//
// Requests arrive as messages are delivered to a maildir: written and synced under a temporary name, then linked to
// their final names (D. J. Bernstein, "Using maildir format"; the manual page maildir(5), "How a message is
// delivered").
//   <spool>/in/<id>.req   A request (lay()): exactly T tokens, each a little-endian 32-bit word
//                         below the vocabulary size V; two more 32-bit words r and e, r < e <= T
//                         (the proof covers positions r to e - 1); and a nonce: 32 bytes from the
//                         system's random generator. <id>: the hash of these bytes (id_of_bytes()).
//   <spool>/work/<id>.req A request that a batch has taken (take()), until its response is on disk.
//   <spool>/out/          A response: <id>.y (the token file: one word) and then <id>.proof, or the
//                         empty file <id>.failed, which does not state the cause. Each file is
//                         written in full under out/.tmp/<id>/ ("staged") and then renamed to its
//                         final name ("named"). <id>.tag ("The tag", below) is created before the
//                         last file, so a response that has <id>.proof or <id>.failed is complete.
//   <spool>/stop          The operator's empty file: the service takes no further batch, removes
//                         the file and exits. Any other entry of that name ends the service too
//                         and is not removed; open() fails while it is there (stop_is_asked()).
//   <spool>/mark.txt      Says that this is a spool directory (check_mark()). The service writes it
//                         at its first start, unless a front end has done so before; nothing
//                         changes or removes it.
//
// Each request taken gets a response (one left in work/: "failed", staged by the next open()),
// unless read_request() says NOT_ITS_NAME in work/. A file in in/ whose id is in out/, out/.tmp/
// or work/ is removed unanswered when take() finds it with a batch slot free (`a_seat_is_left`).
//
// A file <64 hex digits>.req in in/ whose length is not a request's is left there: it may be
// another commitment's request, for which someone waits. So is one that the system does not let
// the service read (EACCES), until it does. One of the right length whose name is not the hash of
// its bytes is removed. Nothing appears in out/ under any of these names. So only someone who
// knows a request's bytes, nonce included, can compute its id. No one else can put a file in its
// place, create or wait for its response's files, or learn from a file of that name whether the
// request exists. The nonce appears in no proof and no file of a response; the tag depends on it.
//
// The directories are for the service's user only: open() fails if the spool directory or one of
// the five in it gives group or others any permission or is another user's (sudo: `typed_sudo`).
// Clients must reach them only through a front end, which this file does not provide. It must
// submit each request itself (lay()), record which client owns the id, and return a response's
// bytes to that owner only. It must show no one a listing of out/, a file's times, or anything
// about an id that is not theirs, and must remove the files of a delivered response.
//
// Release. release() gives the staged responses their final names at once, on the caller's thread.
// Nothing in cp_prove tests a finished proof: a response that is staged is released, and whoever
// hands it on must have verified it.
//
// The tag. <id>.tag is 32 bytes: an HMAC-SHA-256 of the id and the response's files, under a key
// derived from the request's nonce (tag_key_of(), tag_of()). A front end that keeps the key from
// lay() and compares the tag with the files that it is about to return detects a response that
// another account has put under the id. It must pass the tag to no client. Limits: the tag does not
// show that the proof is right, and does not help against anyone who can read the request's file.
// It covers the id and what was staged under it when release() read it. Staged directories swapped
// before that reading get valid tags, and nothing here guards against that. A response that is
// dropped before its release has no tag file. This class zeroes its copies of nonce and key.
//
// One process at a time serves a spool directory, holding locks on it and on out/ while it lives.
// After a kill or a power loss, out/.tmp/ can hold responses that were never released: the next
// open() zeroes and removes them, as does cp_prove -clear (clear()), which starts no service.
// No name or file here states a request's batch slot, its batch's number or the batch's other
// requests; but two proofs of one batch contain the number of the same leaf table (a further table
// of the same weights) and the same shared chain (which all its proofs repeat). A batch's responses
// are named together, in id order, failed or not; how long that takes depends on their number.
//
// Single batch (open_one_batch()): the requests are the files <name>.req of a directory that is
// only read, each exactly T + 2 words, without a nonce; <name> is the id. The responses are staged
// in .tmp of a new directory, or of one that a killed batch has left, which the process locks. No
// tags, no work/, no stop file. The first file in .tmp is a mark.txt of the kind "out", without which neither cp_prove -clear nor a later batch clears anything there.
// In identifiers, "settle" is syncing to disk, "answer" is a response and "lay" is submitting a
// request; "folder" (in messages too) is a directory.
#pragma once
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "common/hmac_sha256.h"
#include "common/walked_path.h" // open_by_parts(), typed_sudo, shown()
#include "weights_commitment/sha256.h"

class Spool : public WalkedPath {
public:
  struct Request {
    std::string id;
    std::vector<uint32_t> tokens; // empty (single batch only): not a request; its slot stays empty
  };
  struct File {
    const char *name;
    const void *bytes;
    size_t n;
  };

  static constexpr size_t NONCE_BYTES = 32;
  // A request's words: T tokens, then r and e. Its file contains them and then the nonce.
  static constexpr size_t words_of(int T) { return (size_t)T + 2; }
  static constexpr size_t bytes_of(int T) { return words_of(T) * 4 + NONCE_BYTES; }
  // A request's id: SHA-256 of a label and the file's bytes (words, then nonce), in lower-case hex.
  static std::string id_of_bytes(const void *bytes, size_t n) {
    Sha256 hash;
    hash.update("p258-spool-req-1", 16);
    hash.update(bytes, n);
    const std::string id = hash.hex();
    explicit_bzero(&hash, sizeof hash); // its buffer may still contain bytes of the nonce
    return id;
  }
  // The key of a response's tag: HMAC-SHA-256, under the request's nonce, of a label.
  static void tag_key_of(const unsigned char nonce[NONCE_BYTES], unsigned char key[32]) {
    HmacSha256 mac(nonce);
    mac.update("p258-spool-tag-1", 16);
    mac.finish(key);
  }
  // The tag: HMAC-SHA-256 under `key` of the id as 32 bytes, the lengths in bytes of y and of the
  // proof, each as a 64-bit little-endian number, then y's bytes and the proof's. For the response
  // "failed" (y and proof null): of the id, sixteen zero bytes and the six letters "failed".
  static void tag_of(const std::string &id, const unsigned char key[32], const std::string *y, const std::string *proof,
                     unsigned char tag[32]) {
    unsigned char head[48] = {0};
    auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    for (size_t i = 0; i < 32 && 2 * i + 1 < id.size(); i++) head[i] = (unsigned char)(digit(id[2 * i]) << 4 | digit(id[2 * i + 1]));
    if (y) lengths_of(*y, *proof, head + 32);
    HmacSha256 mac(key);
    mac.update(head, sizeof head);
    if (y)
      mac.update(y->data(), y->size()), mac.update(proof->data(), proof->size());
    else
      mac.update("failed", 6);
    mac.finish(tag);
  }
  // Submits a request to the spool directory `folder`; a front end calls it. `words` contains the
  // request's words_of(T) words. lay() draws the nonce, writes the file under in/.tmp/, syncs it to
  // disk and renames it without replacing anything, so the file is never seen half written. It
  // returns the id, or "" if a step fails. It takes no lock: any process of the service's user may
  // call it while the spool directory is being served.
  // tag_key: null, or 32 bytes that receive the key of the response's tag, to store with the id.
  static std::string lay(const std::string &folder, const uint32_t *words, int T, unsigned char *tag_key = nullptr) {
    std::vector<unsigned char> bytes(bytes_of(T));
    memcpy(bytes.data(), words, words_of(T) * 4);
    for (size_t have = 0; have < NONCE_BYTES;) {
      const ssize_t n = getrandom(bytes.data() + words_of(T) * 4 + have, NONCE_BYTES - have, 0);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return explicit_bzero(bytes.data(), bytes.size()), "";
      have += (size_t)n;
    }
    if (tag_key) tag_key_of(bytes.data() + words_of(T) * 4, tag_key);
    const std::string id = id_of_bytes(bytes.data(), bytes.size()), name = id + ".req";
    // Every step below names its file relative to these two open directories: what is removed after
    // a failed step is the file that this call created, even if a directory on the path is swapped.
    const int requests = ::open((folder + "/in").c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    const int parts = requests < 0 ? -1 : openat(requests, ".tmp", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    const int fd = parts < 0 ? -1 : openat(parts, id.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    size_t written = 0;
    while (fd >= 0 && written < bytes.size()) {
      const ssize_t n = write(fd, bytes.data() + written, bytes.size() - written);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) break;
      written += (size_t)n;
    }
    bool good = fd >= 0 && written == bytes.size() && !fsync(fd);
    if (fd >= 0) good = !close(fd) && good;
    explicit_bzero(bytes.data(), bytes.size());
    if (good) {
      int named = (int)syscall(SYS_renameat2, parts, id.c_str(), requests, name.c_str(), 1 /* RENAME_NOREPLACE */);
      if (named && (errno == EINVAL || errno == ENOSYS)) { // no such call on this file system: linkat(), then unlinkat()
        named = linkat(parts, id.c_str(), requests, name.c_str(), 0);
        if (!named) unlinkat(parts, id.c_str(), 0);
      }
      good = !named;
    }
    if (!good && fd >= 0) unlinkat(parts, id.c_str(), 0);
    if (parts >= 0) close(parts);
    if (requests >= 0) close(requests);
    if (!good && tag_key) explicit_bzero(tag_key, 32);
    return good ? id : "";
  }

  Spool() {}
  Spool(const Spool &) = delete;
  Spool &operator=(const Spool &) = delete;
  ~Spool() { // closing a directory's descriptor releases its lock
    if (settler.joinable()) { // the thread syncs the responses that still wait, then exits
      {
        std::lock_guard<std::mutex> lock(settle_guard);
        settle_ends = true;
      }
      settle_news.notify_all();
      settler.join();
    }
    for (int fd : {root, in, work, out, tmp})
      if (fd >= 0) close(fd);
  }

  // Opens an existing spool directory (`tokens` is T, `vocabulary` V): takes its first lock; fails,
  // before anything is created or removed, unless the directory has a spool directory's mark.txt or
  // is unused, and marks an unused one (check_mark()); creates missing subdirectories, and clears
  // out/.tmp/ (clear_staged()). For each request left in work/
  // it stages "failed" and withdraws any response in out/: a request leaves work/ only once its
  // response is known to be on disk, so after a power loss that response may be truncated. A file
  // in work/ that read_request() calls NOT_ITS_NAME is only removed. Removals from out/ and work/
  // are synced before it returns. False (`why` set) on error: also if a lock is held, a name is not
  // one that a service leaves, or an entry of work/ is no regular file of a request's length.
  bool open(const std::string &folder, int tokens, uint32_t vocabulary) {
    dir = folder, tmp_path = folder + "/out/.tmp", T = tokens, V = vocabulary;
    // With a vocabulary of 0 every request would be rejected, one by one, with no message.
    if (T <= 0 || !V) return fault("internal error: the spool is opened before the request's length and the vocabulary are known");
    // open() zeroes and removes what it finds staged, so no link and no directory that another
    // account can write may lead it to another directory of the caller's (open_by_parts()). A
    // relative path is rejected: how the caller reached the working directory cannot be checked,
    // and a shell's "cd <another account's link>" makes it a directory of that account's choice.
    std::string refused;
    if (!dir.empty() && dir[0] != '/')
      refused = shown(dir) + " does not begin with /. Name the spool directory by its full path from /: a relative name starts at the working directory, and "
                             "cp_prove cannot tell whether a link led you there (cd follows links)";
    else
      root = dir.empty() ? (errno = ENOENT, -1) : open_by_parts(dir, refused);
    if (!refused.empty()) {
      names_a_part = "the path named with -spool is refused, and nothing was touched: " + refused + ". To start cp_prove with -spool: put the spool "
                     "directory below a directory of your own, in which only you and root can write, and name it by its full path from /";
      return fault(names_a_part.c_str());
    }
    if (root < 0) {
      const int error = errno;
      names_a_part = (dir.empty() ? std::string("an empty name") : shown(dir)) + ", named with -spool, cannot be opened: " + strerror(error) + (error == EACCES ? " (what a run under sudo makes belongs to root: run the service and the client both with sudo, or neither)" : "");
      return fault(names_a_part.c_str());
    }
    // Anyone who can write in the spool directory or a subdirectory can swap a request or a
    // response, or put a link in a subdirectory's place; anyone who can read them sees the
    // requests' ids and when each response appears. So each must pass is_closed_to_others(). The
    // open descriptor is checked: the spool directory's before it is locked, a subdirectory's
    // before anything is created in it (subfolder()).
    if (!is_closed_to_others(root, typed_sudo)) return not_the_users(root, ""); // it may also belong to the account that typed sudo; its subdirectories may not
    if (flock(root, LOCK_EX | LOCK_NB)) return fault(LOCK_IS_HELD);
    // A start that is given the wrong directory, such as another closed directory of the caller's,
    // ends here.
    std::string problem;
    if (const Mark mark = check_mark(root, "spool", true, problem); mark != FITS) {
      const std::string holds = at_the_top(root), another = "the service's spool directory, or a new, empty folder (mkdir -m 700)";
      if (mark == UNREAD || mark == NOT_MADE || mark == NOT_SYNCED)
        names_a_part = shown(dir) + " cannot be used: " + problem + ". Nothing that was in the folder was touched. Fix the cause and start again on the same folder";
      else
        names_a_part = mark_error(dir, "spool", problem) + ". Nothing that was in the folder was touched. " +
                       (mark == MISSING    ? "cp_prove -spool takes a folder that has a spool directory's mark.txt, or an empty one, into which it writes that file. " +
                                                 (holds.empty() ? holds : "This one holds " + holds + ". ") + "Remove nothing to make it fit: name " + another
                        : mark == ILL_LAID ? "Without mark.txt a folder counts as empty if it holds nothing but in/ with waiting requests, an empty out/ and an empty file stop, and in/, in/.tmp "
                                             "and out/ are yours and closed to others (mode 0700). Name " + another + ". Requests that wait in this one, if any, get no response: submit them again"
                                           : "Change nothing in it: name " + another);
      return fault(names_a_part.c_str());
    }
    if ((in = subfolder(root, "in", "/in")) < 0 || (work = subfolder(root, "work", "/work")) < 0 || (out = subfolder(root, "out", "/out")) < 0 ||
        (tmp = subfolder(out, ".tmp", "/out/.tmp")) < 0)
      return false;
    const int parts = subfolder(in, ".tmp", "/in/.tmp"); // for `lay`; only created and checked here
    if (parts < 0) return false;
    close(parts);
    // The second lock stops a cp_prove -clear that is given out/ itself.
    if (flock(out, LOCK_EX | LOCK_NB)) return fault(LOCK_IS_HELD);
    std::vector<std::string> left, held;
    if (!names_in(tmp, left) || !names_in(work, held)) return fault("a subfolder cannot be read");
    // Checked before anything is removed. A service leaves only <id>.req in work/ and <id> in
    // out/.tmp/; a stop file is empty. Other names may be users' files.
    struct stat is;
    bool foreign = !fstatat(root, "stop", &is, AT_SYMLINK_NOFOLLOW) && !(S_ISREG(is.st_mode) && !is.st_size);
    for (const std::string &name : held) foreign = foreign || id_of(name).empty();
    for (const std::string &name : left) foreign = foreign || id_of(name + ".req").empty();
    if (foreign)
      return fault("the folder named with -spool holds a name in work/ that is not <64 hex digits>.req, a name in out/.tmp/ that is not "
                   "<64 hex digits>, or a stop that is not an empty file. Nothing was removed. Name a folder that is used for nothing "
                   "else. Before you give this one up: if a cp_prove has served it, run cp_prove -clear on it: it removes what was staged and never released; if out/.tmp holds an entry whose name cp_prove never uses there, move that entry out first");
    // Also before anything is removed. take() moves only regular files of a request's length to
    // work/. This commitment's service has left no file of another length, directory or link there,
    // and the loop below would remove such a file or link unanswered.
    for (const std::string &name : held)
      if (!fstatat(work, name.c_str(), &is, AT_SYMLINK_NOFOLLOW) && (!S_ISREG(is.st_mode) || is.st_size != (off_t)bytes_of(T)))
        return fault("work/ in the folder named with -spool holds an entry with a request's name that is not a regular file of the length of this commitment's requests, "
                     "4 T + 40 bytes (ls -l shows what each entry is, and its length). cp_prove moves only such files there. Nothing was removed. Name the service's "
                     "spool directory, or, if this is it, a new, empty folder: the requests in in/ and work/ of this one then get no response");
    if (is_a_stop(root)) unlinkat(root, "stop", 0); // a stop file that is already there was meant for an earlier process
    // Staged files are zeroed first: an exit cut short (a kill, a power loss, a lock not taken in
    // time) leaves their bytes unchanged. Only entries with names that cp_prove creates go.
    NotCleared rest;
    clear_staged(tmp, true, tmp_path, rest);
    if (!rest.none())
      return fault("out/.tmp/ holds something that cp_prove did not stage, or that cannot be overwritten or removed. An earlier process's staged "
                   "responses were removed as far as that was possible. Run cp_prove -clear on the folder named with -spool: it lists what is left");
    size_t unanswered = 0;
    for (const std::string &name : held) {
      const std::string id = id_of(name);
      std::vector<uint32_t> tokens;
      const bool a_request = !id.empty() && read_request(work, name, id, tokens) != NOT_ITS_NAME;
      if (!tokens.empty()) explicit_bzero(tokens.data(), tokens.size() * 4);
      if (a_request && !withdraw(id)) return fault("a response that is not known to be on the disk cannot be removed");
      if (!a_request) {
        unanswered += !unlinkat(work, name.c_str(), 0);
        continue;
      }
      if (!stage_failed(id)) return false;
    }
    say_unanswered(unanswered);
    // What was withdrawn or removed above must not come back after a power loss: a response that
    // may be truncated would then be in out/ under its final names.
    if (!held.empty() && (fsync(out) || fsync(work))) return fault("the removals in out/ and work/ cannot be synced to disk");
    opened = true;
    settler = std::thread([this] { settle_for_ever(); });
    return true;
  }

  // Single batch: `requests` is only read; `answers` is created, or taken over from a batch that
  // ended before it released anything. Returns false, with `why` set, if a step fails;
  // a directory that it has created by then stays. mark.txt is the first file in .tmp (check_mark()).
  bool open_one_batch(const std::string &requests, const std::string &answers, int tokens, uint32_t vocabulary) {
    tmp_path = answers + "/.tmp", T = tokens, V = vocabulary, plain = true;
    if (T <= 0 || !V) return fault("internal error: the folders are opened before the request's length and the vocabulary are known");
    in = ::open(requests.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (in < 0) return fault("the folder named with -in does not exist or is not a directory");
    // Trailing slashes are cut: after one, the system follows a link in spite of O_NOFOLLOW.
    const std::string named = answers.substr(0, std::max<size_t>(1, answers.find_last_not_of('/') + 1));
    // The directory in which responses are written is reached as in open(): from /, part by part,
    // so that no link and no directory that another account can write leads the batch elsewhere
    // (open_by_parts()). The walk ends with ENOENT at a name that is not there, after it has checked
    // the directory in which the name is looked up. Then the directory is made, and the walk is
    // repeated.
    std::string refused;
    if (named[0] != '/') refused = (named.empty() ? std::string("an empty name") : shown(named)) + " does not begin with /";
    out = refused.empty() ? open_by_parts(named, refused) : -1;
    const bool missing = out < 0 && errno == ENOENT;
    const bool made = missing && refused.empty() && !mkdir(named.c_str(), 0700);
    if (made) out = open_by_parts(named, refused);
    if (!refused.empty()) {
      names_a_part = std::string("the path named with -out is refused") + (made ? ": " : ", and nothing was touched: ") + refused + ". Put the folder below a directory of your "
                     "own, in which only you and root can write, and name it by its full path from /";
      return fault(names_a_part.c_str());
    }
    if (out < 0 && !made) return fault(missing ? "the folder named with -out cannot be created" : NOT_OPENED);
    // What is opened must look like what was created: this user's, closed to group and others.
    if (out < 0 || !is_closed_to_others(out) || flock(out, LOCK_EX | LOCK_NB)) // the lock stops a cp_prove -clear while this process lives
      return fault(made ? "the folder named with -out was created, but what now has that name cannot be opened or locked, is a link, is not yours, or is open "
                          "to your group or to others. Nothing was written into it. Name a new folder on a file system that keeps owners and modes, in a folder "
                          "that you alone can write"
                        : EXISTS);
    // A directory that exists is taken only as a batch leaves it that was killed before it released
    // anything: it holds nothing but .tmp, and .tmp has a mark.txt of the kind "out"
    // and only names that cp_prove stages. Until all of that is known, nothing in it is touched or
    // written: no mark is made here, so a spool directory's out/, whose .tmp has none, is refused.
    std::vector<std::string> top, staged;
    if (!made && !(names_in(out, top, true) && top == std::vector<std::string>{".tmp"})) return fault(EXISTS);
    if (made && mkdirat(out, ".tmp", 0700)) return fault("a subfolder cannot be created");
    tmp = openat(out, ".tmp", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (tmp < 0) return fault(made ? "a subfolder cannot be opened" : EXISTS);
    if (!made && !(is_closed_to_others(tmp) && names_in(tmp, staged, true) && std::all_of(staged.begin(), staged.end(), [](const std::string &name) {
          return is_mark_name(name) || lays_a_folder(STAGING, name) == RESPONSE;
        })))
      return fault(EXISTS);
    std::string problem;
    if (check_mark(tmp, "out", made, problem) != FITS) {
      if (!made) return fault(EXISTS);
      names_a_part = "the new folder named with -out cannot be used: " + problem + ". The folder was created and holds no response: name another";
      return fault(names_a_part.c_str());
    }
    if (!made) { // what the earlier batch staged was never released: it goes, as in clear(); what cannot go stays, and the call fails
      NotCleared rest;
      clear_staged(tmp, false, tmp_path, rest);
      if (!rest.none()) return fault("the folder named with -out holds what an earlier batch staged, and not all of it could be removed. Run cp_prove -clear on that folder: it lists what is left");
      took_over = !std::all_of(staged.begin(), staged.end(), is_mark_name);
    }
    opened = true;
    settler = std::thread([this] { settle_for_ever(); });
    return true;
  }
  // Lists the names <name> of the files <name>.req in a directory, in bytewise order: the requests
  // of a single batch, slot by slot. A name has 1 to 64 characters out of A-Z a-z 0-9 _ -. Every
  // program must agree on which names count, so any other name, and any entry that is not a regular
  // file, is an error [cp-in-name], and `odd` receives its name. False then, or on a read error.
  static bool requests_in(const std::string &folder, std::vector<std::string> &names, std::string &odd) {
    names.clear(), odd.clear();
    const int fd = ::open(folder.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    DIR *d = fd < 0 ? nullptr : fdopendir(fd);
    if (!d) {
      if (fd >= 0) close(fd);
      return false;
    }
    const struct dirent *e;
    for (errno = 0; odd.empty() && (e = readdir(d)); errno = 0) {
      const std::string file = e->d_name;
      if (file == "." || file == "..") continue;
      bool fit = file.size() >= 5 && file.size() <= 68 && !file.compare(file.size() - 4, 4, ".req");
      for (size_t c = 0; fit && c + 4 < file.size(); c++) {
        const unsigned char x = (unsigned char)file[c];
        fit = (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z') || (x >= '0' && x <= '9') || x == '_' || x == '-';
      }
      struct stat is;
      fit = fit && !fstatat(fd, file.c_str(), &is, AT_SYMLINK_NOFOLLOW) && S_ISREG(is.st_mode);
      if (fit)
        names.push_back(file.substr(0, file.size() - 4));
      else
        odd = file;
    }
    const bool whole = odd.empty() && !errno;
    closedir(d); // it closes fd
    std::sort(names.begin(), names.end());
    return whole;
  }
  // For the service's exit, after settled(): removes a single batch's empty .tmp.
  void tidy() const {
    if (!opened) return;
    remove_the_mark();
    if (plain && !unlinkat(out, ".tmp", AT_REMOVEDIR)) staging_is_gone = true;
  }
  // Single batch: mark.txt is removed only when .tmp is otherwise empty, because cp_prove -clear
  // requires it. Call it when nothing more can be staged.
  void remove_the_mark() const {
    std::vector<std::string> left;
    if (plain && names_in(tmp, left, true) && std::all_of(left.begin(), left.end(), is_mark_name))
      for (const std::string &name : left) unlinkat(tmp, name.c_str(), 0);
  }
  // Any entry named stop ends the service: to ignore a stop that was meant would cost more than to
  // obey one that was not. Only an empty regular file is removed; anything else makes open() fail.
  bool stop_is_asked() const { return !plain && is_there(root, "stop"); }
  // After a stop: true if no entry named stop is left. One that is not an empty regular file stays,
  // reported on stdout (/dev/null in a production build: the first notice is a failing open()).
  bool stop_is_obeyed() const {
    if (plain || !is_there(root, "stop")) return true;
    if (is_a_stop(root)) return !unlinkat(root, "stop", 0);
    printf("the spool: the entry named stop in the spool directory has ended the service. It is not an empty regular file, so it was left where it "
           "is. Find out what put it there. While it is there, cp_prove -spool does not start in that directory\n");
    return false;
  }
  // The number of staged responses that wait for release().
  size_t staged() {
    std::lock_guard<std::mutex> lock(guard);
    return waiting.size();
  }
  static constexpr int REFUSALS_MOST = 1024;   // bounds what one `take` reads, rejects and removes
  size_t removed_unanswered = 0;               // the sum of `say_unanswered`'s counts so far
  size_t unfit_told = 0;                       // wrong-length files that the last `take` counted
  size_t unreadable = 0;                       // files that the last `take` was not allowed to read
  size_t LOOK_MOST = 100000;                   // most right-length files per scan; tests set less
  // now: how many entries the calling open() or take() has removed because read_request() said
  // NOT_ITS_NAME. They get no response: a response may appear under an id only for someone who knew
  // the request's bytes, and its tag's key is derived from them. The printed line makes the removal
  // visible. A production build shows only open()'s line: stdout is /dev/null before any take().
  void say_unanswered(size_t now) {
    removed_unanswered += now;
    if (now)
      printf("the spool: files with a request's name and length whose name is not the hash of their bytes, or that could not be read: %zu. They "
             "were removed and get no response: whoever waits for one must submit the request again\n", now);
  }

  // Takes the oldest requests of in/, at most `most`, by modification time and then by name. Left
  // alone: a name that is not a request's, an entry that is no regular file, a file whose length is
  // not a request's, a file that the system refuses (EACCES) to open for reading (`unreadable`).
  // A file whose name is not the hash of its bytes is removed without a response.
  // A request with a token not below V, or without r < e <= T, takes no batch slot: "failed" is
  // staged for it. `taken` is sorted by id. Returns its size, or -1, with `why` set, on a fault.
  //
  // Any number of files can arrive in in/, so the work of one call is bounded:
  //  - A scan that stops at LOOK_MOST right-length files sets `cut_short`: those seen need not be
  //    the oldest, and in/ may hold more even if none was taken. A call takes or removes some of
  //    what it saw, so in/ shrinks. Other names stay and slow every scan: in/ is for requests only.
  //  - Files of a wrong length stay too. They compete for no batch slot and do not count against
  //    LOOK_MOST, and a printed line gives their number: a front end should check lengths first.
  //  - So it is with files that the system refuses (EACCES) to open, such as one that root has written with mode 0600
  //    for a service that runs as another account. None of its bytes is known, so it is neither
  //    moved nor removed, and nothing is answered under its name. Each scan tries again, at the
  //    cost of one refused open, and takes it like any other file once it may be read. Any other file of the right length that cannot be read, this user's own with the read bit among them, is moved to work/ and removed without a response; if its id already has a response or is being served, it is removed from in/ instead.
  //  - Of the files of the right length, only the oldest `most` + REFUSALS_MOST are read. (The scan
  //    opens and closes one that is not this user's own or lacks the read bit; it reads nothing.)
  //    Those that cannot be served are dealt with at once, even when every batch slot is filled. If
  //    slots stay free and not all were read, `left_unopened` is set: in/ may still hold requests.
  bool cut_short = false, left_unopened = false; // set by the last take()
  int take(std::vector<Request> &taken, int most) {
    taken.clear();
    cut_short = left_unopened = false;
    if (settle_failed) return fault("a response could not be synced to disk"), -1;
    if (plain) return take_the_folder(taken, most);
    struct Waiting {
      struct timespec written;
      std::string name;
      bool operator<(const Waiting &b) const {
        if (written.tv_sec != b.written.tv_sec) return written.tv_sec < b.written.tv_sec;
        if (written.tv_nsec != b.written.tv_nsec) return written.tv_nsec < b.written.tv_nsec;
        return name < b.name;
      }
    };
    std::vector<Waiting> fit; // files of the right length
    size_t unfit = 0;         // how many have a wrong one
    size_t denied = 0;        // how many of the right length may not be read
    std::string a_denied;     // of these names the first in bytewise order
    // Whether the system refuses to open the regular file `name` in in/ to read it; `s`: its lstat.
    // Only a refused open (EACCES) says yes. This user's own file with the owner's read bit is not
    // tried; if it cannot be read after all, read_request() fails in work/ and it is removed.
    const uid_t user = geteuid();
    auto is_denied = [&](const struct stat &s, const char *name) {
      if (s.st_uid == user && (s.st_mode & S_IRUSR)) return false;
      const int fd = openat(in, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
      if (fd >= 0) close(fd);
      return fd < 0 && errno == EACCES;
    };
    {
      const int again = openat(in, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
      DIR *d = again < 0 ? nullptr : fdopendir(again);
      if (!d) {
        if (again >= 0) close(again);
        return fault("in/ cannot be read"), -1;
      }
      // Only names that a call can take or remove count against LOOK_MOST. A name that stays would
      // be read again by every scan, and a scan filled with such names would never find a request.
      // readdir() returns null at the end of the listing and on an error; only an error sets errno.
      const struct dirent *e;
      for (errno = 0; (e = readdir(d)); errno = 0) { // fstatat() below can set errno too
        if (fit.size() == LOOK_MOST) {
          cut_short = true;
          break;
        }
        struct stat s;
        if (id_of(e->d_name).empty() || fstatat(in, e->d_name, &s, AT_SYMLINK_NOFOLLOW) || !S_ISREG(s.st_mode)) continue;
        if (s.st_size != (off_t)bytes_of(T))
          unfit++;
        else if (!is_denied(s, e->d_name))
          fit.push_back({s.st_mtim, e->d_name});
        else if (denied++; a_denied.empty() || a_denied.compare(e->d_name) > 0)
          a_denied = e->d_name;
      }
      const bool whole = e || !errno; // e is not null if the scan was cut short
      closedir(d);
      if (!whole) return fault("in/ cannot be read"), -1;
    }
    bool moved = false; // whether this call has moved a request
    // Moves a request to work/. Returns 1 if it did; 0 if the request is not to be served (its id
    // has a response or is being served, or the submitter has removed the file); -1 on a fault.
    auto claim = [&](const std::string &name, const std::string &id) {
      if (is_answered(id) || is_there(tmp, id) || is_there(work, name)) return unlinkat(in, name.c_str(), 0), 0;
      if (!renameat(in, name.c_str(), work, name.c_str())) {
        moved = true;
        return 1;
      }
      return errno == ENOENT ? 0 : (fault("a request cannot be moved from in/ to work/"), -1);
    };
    size_t unanswered = 0;
    const size_t opened_most = std::min(fit.size(), (size_t)most + REFUSALS_MOST);
    std::partial_sort(fit.begin(), fit.begin() + opened_most, fit.end());
    int refused = 0;
    for (size_t i = 0; i < opened_most; i++) {
      const bool a_seat_is_left = (int)taken.size() < most;
      if (!a_seat_is_left && refused >= REFUSALS_MOST) break;
      Request r;
      r.id = id_of(fit[i].name);
      // When every batch slot is filled, a request waits in in/ for the next batch. A file that
      // cannot be served need not wait, and reading it in in/ shows which of the two it is.
      if (!a_seat_is_left && read_request(in, fit[i].name, r.id, r.tokens) == A_REQUEST) {
        explicit_bzero(r.tokens.data(), r.tokens.size() * 4);
        continue;
      }
      const int claimed = claim(fit[i].name, r.id);
      if (claimed < 0) return -1;
      if (!claimed) continue;
      Read read = read_request(work, fit[i].name, r.id, r.tokens); // this read decides, not the one in in/
      if (read == A_REQUEST && a_seat_is_left) {
        taken.push_back(std::move(r));
        continue;
      }
      if (read == A_REQUEST) { // the read in in/ met a fault or the file changed since; no slot left: it gets "failed"
        explicit_bzero(r.tokens.data(), r.tokens.size() * 4);
        read = A_TOKEN_OUTSIDE;
      }
      refused++;
      if (read == NOT_ITS_NAME)
        unanswered += !unlinkat(work, fit[i].name.c_str(), 0); // not a request, or its read failed: nothing appears in out/ under its name
      else if (!stage_failed(r.id))
        return -1;
    }
    say_unanswered(unanswered);
    // Printed when `unfit` changes. A production build's stdout is /dev/null: only in/ shows it.
    if (unfit && unfit != unfit_told)
      printf("the spool: files in in/ that have a request's name but not the length of this commitment's requests: %zu. They get no response and "
             "stay where they are. Look at them: they may be another commitment's requests\n", unfit);
    unfit_told = unfit;
    // Printed when `denied` changes and is not 0. A production build's stderr is /dev/null too:
    // there the service prints one constant line, once (prover/prover/service_loop.h).
    if (denied && denied != unreadable)
      fprintf(stderr, "the spool: files in in/ that have a request's name and length and that cp_prove may not read (%s): %zu, among them %s. They stay where they are and get no "
                      "response. ls -l shows their owners and modes: what a run as root or under sudo writes belongs to root. Each is taken like any other file once the account that runs "
                      "cp_prove may read it\n", strerror(EACCES), denied, a_denied.c_str());
    unreadable = denied;
    // Else a caller that exits when in/ is empty could miss a request younger than those opened.
    left_unopened = (int)taken.size() < most && opened_most < fit.size();
    // The moves must be on disk before the caller begins a batch: a power loss could undo one, and
    // the next process would take the request into a second batch. work/ is synced first: if only
    // that sync is on disk, open() stages "failed" for the request and claim() removes it from in/.
    if (moved && (fsync(work) || fsync(in))) return fault("the moves from in/ to work/ cannot be synced to disk"), -1;
    // Sorted by id: a request's batch slot then depends only on which requests are taken together.
    std::sort(taken.begin(), taken.end(), [](const Request &a, const Request &b) { return a.id < b.id; });
    return (int)taken.size();
  }

  // Appends parts to a response's files "y" and "proof" (File::name; not checked) in
  // out/.tmp/<id>/, which it creates if missing. Call it whenever parts are ready, from one thread
  // at a time. It writes nothing and returns true if the response is staged as "failed". A call
  // that begins after shut() or drop_what_is_staged() writes nothing and returns false.
  bool stage_answer(const std::string &id, const std::vector<File> &files) {
    const long long asked = steady_us();
    std::lock_guard<std::mutex> lock(guard);
    locking_us += steady_us() - asked;
    Answer *answer = note(id);
    if (!answer) return false;
    return answer->failed || write_files(id, files);
  }
  // Stages "failed" in place of what is staged for the id, which is first overwritten with zeros.
  bool stage_failed(const std::string &id) {
    const long long asked = steady_us();
    std::lock_guard<std::mutex> lock(guard);
    locking_us += steady_us() - asked;
    Answer *answer = note(id);
    if (!answer) return false;
    const long long begun = steady_us();
    const bool staged = turn_to_failed(id, *answer);
    making_us += steady_us() - begun;
    return staged;
  }

  // Gives every staged response's files their final names, in the order of the ids, and queues the
  // ids for the syncing thread (`settler`). If time_ms is not negative, each file first gets it as
  // modification and access time: its own times would show how long each response took. The
  // status-change time becomes the renaming's and the birth time cannot be set, so a front end must
  // pass on only the files' bytes. A file whose final name is taken is dropped. False, with `why`
  // set, if another step fails; unnamed responses stay in out/.tmp/.
  bool release(long long time_ms) {
    if (settle_failed) return fault("a response could not be synced to disk");
    std::lock_guard<std::mutex> lock(guard);
    if (closed) return fault("the process is ending");
    const long long begun = steady_ms();
    for (auto &[id, answer] : waiting) // each response is read once, to compute its tag
      if (!plain && !answer.failed && !tagged(id, answer)) return fault("a staged response cannot be read");
    if (!name_them(waiting, time_ms)) return false;
    waiting.clear();
    naming_ms = steady_ms() - begun;
    return true;
  }
  // Waits until every released response is on disk and its request has left work/. Waits at most
  // wait_ms milliseconds, or without limit if wait_ms is negative. Returns true if that state was
  // reached. False, with `why` set, also if the syncing thread has failed.
  bool settled(long long wait_ms) {
    std::unique_lock<std::mutex> lock(settle_guard);
    auto idle = [&] { return named.empty() && !settling; };
    if (wait_ms < 0)
      settle_news.wait(lock, idle);
    else
      settle_news.wait_for(lock, std::chrono::milliseconds(wait_ms), idle);
    if (settle_failed) return fault("a response could not be synced to disk");
    return idle();
  }
  // Durations, for measurements: the three parts of the last release; and, summed over all threads
  // since they were last set to 0, the time spent writing and syncing the responses' files.
  long long naming_ms = 0;
  std::atomic<long long> syncing_ms{0}, removing_ms{0}; // the syncing thread's last round: sync of out/; removals and sync of work/
  std::atomic<long long> writing_us{0}, syncing_us{0};
  std::atomic<long long> making_us{0}, locking_us{0}; // create directories, files; wait for `guard`
  std::atomic<int> slow_disk_ms{0}; // for tests only: the syncing thread sleeps this long before each round's syncs
  // Milliseconds since the epoch (CLOCK_REALTIME): the time that is set on a batch's responses.
  static long long now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (long long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  }
  static long long steady_us() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000 + t.tv_nsec / 1000;
  }
  static long long steady_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  }

  // Only sets the flag `closed`: no lock, no allocation, no wait. The service calls it at the start
  // of an exit (g_end_first in prover/prover/ends.h) and when a self-check on secret random values
  // fails (a_self_check_failed() in prover/prover/prover.h). No later call stages anything, and no
  // batch whose naming has not begun is named: release() reads the flag under `guard` before it
  // names. A naming under way is finished, so shut() splits no batch. Each response of that batch
  // that is not "failed" was staged before the flag was set.
  // The fence keeps the flag ahead of the caller's later stores (keys and masks zeroed).
  void shut() {
    closed = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  // Removes what is staged and releases no response; no later call stages anything. For the
  // process's exit, from any thread. The requests stay in work/, and the next process answers them
  // "failed". A response's files are zeroed before they are unlinked. It waits about two seconds for
  // the lock: the holder may be stuck in a system call forever. False if this or an earlier call got
  // no lock or could not zero or remove a file: bytes may stay staged until a later open() or clear(),
  // or stay on disk, beyond their reach, where erase() unlinked a file that it could not zero.
  bool drop_what_is_staged() {
    closed = true;
    if (!opened) return true; // no opening has succeeded: no file is touched (open() may have staged "failed"; the next open() removes it)
    std::unique_lock<std::mutex> lock(guard, std::defer_lock);
    for (int i = 0; i < 200 && !lock.try_lock(); i++) usleep(10000);
    if (!lock) return !(left_as_it_was = true);
    bool all = true;
    for (const auto &[id, answer] : waiting) all = erase(id) && all;
    waiting.clear();
    NotCleared rest;
    if (!staging_is_gone) clear_staged(tmp, !plain, tmp_path, rest); // after a good single batch tidy() has removed it
    if (!all || !rest.none()) left_as_it_was = true; // stays set for later calls
    remove_the_mark(); // `closed` is set and `guard` is held: an end with nothing staged leaves an empty .tmp
    return !left_as_it_was;
  }

  // ---- clearing what an earlier process staged ----
  // After a kill or a loss of power the staging directory (out/.tmp in a spool directory, .tmp in
  // the -out directory) can hold responses that were never released. cp_prove creates only these
  // entries there:
  //   <a response's name>/ with the files y, proof and failed. In a spool directory the name is the
  //     request's id, 64 lower-case hex digits (id_of()); for a single batch it is the request
  //     file's name, 1 to 64 characters out of A-Z a-z 0-9 _ - (requests_in());
  //   for a single batch, mark.txt (is_mark_name()), which is kept and not reported.
  // Code that creates any further name there must add it to lays_a_file() or lays_a_folder(). What
  // clear_staged() leaves is reported in a NotCleared.
  struct NotCleared {
    struct Refused {
      std::string path;
      const char *step; // what failed: "REMOVED", "READ", "OVERWRITTEN" or "SYNCED TO THE DISK"
      int error;        // the errno that the step set
    };
    std::vector<std::string> foreign; // paths of entries that cp_prove does not create: not touched
    std::vector<Refused> refused;
    bool none() const { return foreign.empty() && refused.empty(); }
    void refuse(const std::string &path, const char *step) {
      const int error = errno; // saved first: the allocations below can change errno
      refused.push_back({path, step, error});
    }
  };
  // Removes the entries listed above by name, follows no link, and reports every other entry
  // without touching it; open() then fails. A response's files may not have been checked, so each
  // is overwritten with zeros and synced before it is removed (overwrite_file()). One that cannot
  // be overwritten stays and is reported; so does one with a second name, untouched. A disk may
  // still keep the bytes. Below the staging directory no directory's owner or mode is checked;
  // overwrite_file() writes only to this user's regular files. A step that it or the system
  // rejects is reported with its errno.
  // tmp: the open staging directory; the caller must hold the lock on its parent directory.
  // in_a_spool: it is a spool directory's out/.tmp, so only an id names a response there: an exit
  // of a running service must not take a directory that someone has put there under another name
  // for a response's. path: the staging directory's path, with which the reported paths begin. The
  // staging directory itself stays. It is synced to disk, and so is each of cp_prove's directories
  // that stays because an entry in it stays.
  static void clear_staged(int tmp, bool in_a_spool, const std::string &path, NotCleared &rest) {
    clear_in(tmp, in_a_spool ? SPOOL_STAGING : STAGING, path, rest);
    if (fsync(tmp)) rest.refuse(path, "SYNCED TO THE DISK");
  }

  // cp_prove -clear <folder>, except for the printing. folder: the directory that was named with
  // -spool or -out; "" stands for a wrong number of arguments. Clears <folder>/out/.tmp and
  // <folder>/.tmp, whichever exist (clear_staged()), and creates nothing.
  // Before anything is touched in either:
  //  - the named path begins with /, as in open(), and is walked part by part (open_by_parts()): no
  //    part may be a link, and no one but root, the caller and `typed_sudo` may be able to swap one;
  //  - <folder>, out/ and each staging directory are directories and no links, this user's
  //    (<folder> alone may be `typed_sudo`'s), and closed to group and others, as open() and
  //    open_one_batch() create and require them;
  //  - the locks on <folder> and on its out/, which a living cp_prove holds, are free. They are not
  //    waited for;
  //  - each staging directory has its mark.txt (check_mark()): <folder>/mark.txt
  //    of the kind "spool" for out/.tmp, .tmp/mark.txt of the kind "out" for .tmp. Only an empty
  //    staging directory may lack the file: it is left as it is;
  //  - each name at the top of a staging directory is one that cp_prove creates there: in out/.tmp,
  //    as in open(), 64 lower-case hex digits; in .tmp, see the head of this section.
  // So it never clears under a living process, no account but those three can lead it elsewhere,
  // and in another closed directory of the caller's, named by mistake, nothing goes.
  // Limits. In a marked directory it knows what cp_prove staged by names alone, and it clears any
  // commitment's: what goes there was staged and never released.
  // said: receives the lines to print. They do not say what was removed: how many proofs are staged
  // at a kill can depend on which requests had failed.
  // Returns the exit status. 0: nothing that cp_prove staged is left. 1: a person must act, and the
  // lines say how; also for a wrong call. 2: nothing was touched and nothing is known to be wrong:
  // a cp_prove runs there, there is no such directory, or it has no staging directory.
  static int clear(const std::string &folder, std::string &said) {
    struct Folder {
      int fd = -1;
      ~Folder() {
        if (fd >= 0) close(fd); // this also releases the lock on it, if there is one
      }
    };
    auto say = [&said](const std::string &line) { said += "cp_prove -clear: " + line + "\n"; };
    if (folder.empty()) {
      say("usage: cp_prove -clear <folder>, with no other argument. <folder> is the folder that was named with -spool or with -out. Nothing was "
          "touched.");
      return 1;
    }
    if (folder[0] != '/') {
      say("nothing was touched: " + shown(folder) + " does not begin with /. Name the folder by its full path from /: a relative name starts at the working "
          "directory, and cp_prove cannot tell whether a link led you there (cd follows links).");
      return 1;
    }
    // `top` is only for the printed lines: they show it, and whose() looks at it for their words. A
    // path that cannot be resolved is shown as it was given.
    char *const resolved = realpath(folder.c_str(), nullptr);
    const std::string top = resolved ? resolved : folder;
    free(resolved);
    // Each of the next three lambdas returns 0, or the exit status with which clear() ends.
    auto examine = [&](int fd, int error, const std::string &path, uid_t or_of = 0) { // error: the errno of the open that gave fd; or_of: as in `is_closed_to_others`
      if (fd < 0 && (error == ELOOP || error == ENOTDIR)) {
        say("nothing was touched: " + shown(path) + " is a link or a file, and cp_prove makes only a folder under that name. If another folder was "
            "named with -spool or -out, name that one. If it was this one, look at it (ls -ld) and give nothing from there to anyone.");
        return 1;
      }
      if (fd < 0) {
        say("nothing was touched: " + shown(path) + " cannot be opened: " + strerror(error) + (error == EACCES ? ". " + whose(path) + (or_of ? "The named folder may also belong to user " + std::to_string((unsigned long)or_of) + ", who typed sudo. " : std::string()) + (geteuid() ? "If the folder is yours and only its mode is closed, give yourself its permissions back (chmod 700) and run this again. If not, run" : "Run") + " every step on one folder the same way, with sudo or without: what a run as root or under sudo makes belongs to root." : ". Fix the cause and run this again."));
        return 1;
      }
      if (!is_closed_to_others(fd, or_of)) {
        say("nothing was touched: " + shown(path) + " belongs to another user, or is open to its group or to others. cp_prove stages responses only "
            "in folders of its own user's with mode 0700. If another folder was named with -spool or -out, name that one. If it was this one, "
            "give nothing from there to anyone. If it is another user's, run this as the user who ran cp_prove. If it is yours and only the mode "
            "is open, find out who could write there; then, unless you are root, close it (chmod 700) and run this again; as root, give nothing "
            "from it to anyone and start the next service on a new folder made with mode 700.");
        return 1;
      }
      return 0;
    };
    auto lock = [&](int fd, const std::string &path) {
      if (!flock(fd, LOCK_EX | LOCK_NB)) return 0;
      const int error = errno;
      if (error != EWOULDBLOCK) {
        say("nothing was touched: " + shown(path) + " cannot be locked: " + strerror(error) + ". Fix the cause and run this again.");
        return 1;
      }
      say("nothing was touched: a cp_prove is running on " + shown(path) + ". Run this after it has ended.");
      return 2;
    };
    auto below = [&](int parent, const char *name, const std::string &path, Folder &found) { // found.fd stays -1 if there is no such name
      found.fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      const int error = errno; // saved at once: string operations can change errno
      return found.fd < 0 && error == ENOENT ? 0 : examine(found.fd, error, path);
    };
    const std::string paths[2] = {top + "/out/.tmp", top + "/.tmp"};
    Folder root, out, staging[2];
    std::string refused;
    root.fd = open_by_parts(folder, refused);
    const int error = errno;
    if (!refused.empty()) {
      say("nothing was touched: " + refused + ".");
      return 1;
    }
    if (root.fd < 0 && error == ENOENT) {
      say("nothing was touched: there is no folder " + shown(top) + ". Name the folder that was named with -spool or -out, by its absolute path.");
      return 2;
    }
    if (int status = examine(root.fd, error, top, typed_sudo)) return status; // as in `open`: the named directory itself may belong to `typed_sudo`
    if (int status = lock(root.fd, top)) return status;
    if (int status = below(root.fd, "out", top + "/out", out)) return status;
    if (out.fd >= 0) {
      if (int status = lock(out.fd, top + "/out")) return status;
      if (int status = below(out.fd, ".tmp", paths[0], staging[0])) return status;
    }
    if (int status = below(root.fd, ".tmp", paths[1], staging[1])) return status;
    if (staging[0].fd < 0 && staging[1].fd < 0) {
      say("nothing was touched: there is no " + shown(paths[0]) + " and no " + shown(paths[1]) + ", the two places where cp_prove stages responses. If " +
          shown(top) + " is the folder that was named with -spool or -out, nothing is left to clear.");
      return 2;
    }
    // Both staging directories pass both checks, the mark and the names, before anything is removed.
    bool marked[2] = {false, false};
    std::vector<std::string> odd; // names at the top of a staging directory that cp_prove never creates there, as paths
    for (int i = 0; i < 2; i++) {
      if (staging[i].fd < 0) continue;
      std::vector<std::string> staged;
      if (!names_in(staging[i].fd, staged, true)) {
        const int unread = errno;
        say("nothing was touched: " + shown(paths[i]) + " cannot be read: " + strerror(unread) + ". Fix the cause and run this again.");
        return 1;
      }
      for (const std::string &name : staged) // out/.tmp: as in open()
        if (!(i ? is_mark_name(name) || lays_a_folder(STAGING, name) == RESPONSE : !id_of(name + ".req").empty())) odd.push_back(paths[i] + "/" + name);
      std::string problem;
      const Mark mark = check_mark(i ? staging[1].fd : root.fd, i ? "out" : "spool", false, problem);
      if ((marked[i] = mark == FITS)) continue;
      if (mark == MISSING && (i ? std::all_of(staged.begin(), staged.end(), is_mark_name) : staged.empty())) {
        say("nothing is staged under " + shown(paths[i]) + ", and no mark.txt says that it is cp_prove's: it was left as it is. If another folder was named with -spool or "
            "-out, name that one.");
        continue;
      }
      if (mark == UNREAD) {
        say("nothing was touched: " + shown(i ? paths[1] : top) + " cannot be examined: " + problem + ". Fix the cause and run this again. Until then give nothing from " + shown(paths[i]) + " to anyone.");
        return 1;
      }
      say("nothing was touched: " + mark_error(i ? paths[1] : top, i ? "out" : "spool", problem) + ". cp_prove -clear removes nothing unless mark.txt says that the folder is "
          "cp_prove's (cp_prove -spool writes that file into its folder, cp_prove -out into .tmp). " +
          (i && marked[0] ? shown(top) + " is a marked spool directory, at whose top cp_prove makes no .tmp. Give nothing from that .tmp to anyone. Look at it, move it out of the folder and run this again: until then " +
                                shown(paths[0]) + " is not cleared either."
                          : (i ? "If " + shown(top) + " is out/ of a spool directory, name the spool directory. " : std::string()) + "If another folder was named with -spool or -out, name that one. "
                            "If it was this one, give nothing from " + shown(paths[i]) + " to anyone, now or later: responses that were staged and never released may be in it."));
      return 1;
    }
    if (!marked[0] && !marked[1]) return 0; // each staging directory that exists is empty
    if (!odd.empty()) {
      constexpr size_t LINES_MOST = 20;
      std::sort(odd.begin(), odd.end());
      for (size_t k = 0; k < odd.size() && k < LINES_MOST; k++) say("LEFT, not touched: " + shown(odd[k]));
      if (odd.size() > LINES_MOST) say("And more is LEFT, not touched.");
      say("nothing was touched, and what cp_prove staged there stays: the entries listed as LEFT have names that cp_prove never uses in a staging folder. If " + shown(top) +
          " is not the folder that was named with -spool or -out, change nothing in it and name that folder. If it is, give none of them to anyone: each may be a copy of a staged "
          "response. Look at each; when you know what it is, move it out of this folder and run this again.");
      return 1;
    }
    int status = 0;
    for (int i = 0; i < 2; i++) {
      if (!marked[i]) continue;
      NotCleared rest;
      clear_staged(staging[i].fd, i == 0, paths[i], rest);
      if (rest.none()) {
        say("nothing that cp_prove staged is left under " + shown(paths[i]) + ".");
        continue;
      }
      status = 1;
      constexpr size_t LINES_MOST = 20;
      for (size_t k = 0; k < rest.refused.size() && k < LINES_MOST; k++)
        say(std::string("COULD NOT BE ") + rest.refused[k].step + ": " + shown(rest.refused[k].path) + ": " + strerror(rest.refused[k].error) + ".");
      if (rest.refused.size() > LINES_MOST) say("And more could not be removed, read, overwritten or synced.");
      if (!rest.refused.empty())
        say("So something that cp_prove staged, and did not release, may still be under " + shown(paths[i]) + ": give nothing from there to anyone. "
            "Fix the cause and run this again.");
      for (size_t k = 0; k < rest.foreign.size() && k < LINES_MOST; k++) say("LEFT, not touched: " + shown(rest.foreign[k]));
      if (rest.foreign.size() > LINES_MOST) say("And more is LEFT, not touched.");
      if (!rest.foreign.empty())
        say("cp_prove did not make the entries listed as LEFT, or someone gave them a second name. If " + shown(top) + " is not the folder that was "
            "named with -spool or -out, change nothing in it and name that folder. If it is, give none of them to anyone: each may be a copy of a "
            "staged response. Look at each; when you know what it is, move it out of this folder and run this again." +
            (i ? "" : " cp_prove -spool does not start on this folder while they are there."));
    }
    return status;
  }

  // ---- mark.txt ----
  // Says what kind of directory this is, so that a call that is given the wrong directory stops
  // before it changes anything there. It guards against mistakes, not against someone who can write
  // there. Its third line is 64 hex digits: cp_prove knows no commitment ID, writes 64 zeros, and
  // takes any 64 digits that it finds. integrations/spool.py writes the same form with a commitment
  // ID of its own, and takes 64 zeros as any commitment's.
  static constexpr const char *MARK_FILE = "mark.txt";
  // The file is written under a name with this prefix and then renamed. A kill can leave such a
  // name. It counts as nothing, and in a spool directory nothing removes it.
  static constexpr const char *MARK_TMP_PREFIX = "mark.txt.making.";
  // A response's name has no dot, so these names never collide with a response's directory in .tmp.
  static bool is_mark_name(const std::string &name) { return name == MARK_FILE || !name.compare(0, strlen(MARK_TMP_PREFIX), MARK_TMP_PREFIX); }
  static bool is_commitment_id(const std::string &id) { return id.size() == 64 && id.find_first_not_of("0123456789abcdef") == std::string::npos; }
  static std::string mark_text(const char *kind) { return std::string("cp-folder 1\nkind ") + kind + "\ncommitment " + std::string(64, '0') + "\n"; }
  enum Mark {   // what check_mark() finds
    FITS,       // a regular file, not a link, with the bytes of mark_text() but for any 64 hex digits
    MISSING,    // no entry of that name
    ILL_LAID,   // the same, in a directory that would be unused but for a fault in in, out or stop
    WRONG,      // anything else of that name
    UNREAD,     // it cannot be opened or read; it is missing and the directory cannot be listed; an internal error
    NOT_MADE,   // it was missing, and a step of create_mark() failed
    NOT_SYNCED, // it is there, but the directory could not be synced to disk
  };
  // Checks mark.txt in the open directory dir_fd: the top of a spool directory (kind "spool"), or
  // .tmp of a single batch's -out directory (kind "out"). problem: receives words for a message.
  // create: if the file is missing and the directory is unused, write it and read it back. Unused:
  // it holds only unfinished marks and, for "spool", what a_client_puts_first() accepts.
  static Mark check_mark(int dir_fd, const char *kind, bool create, std::string &problem) {
    for (bool made = false;; made = true) {
      const int fd = openat(dir_fd, MARK_FILE, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
      int error = errno; // saved at once: string operations can change errno
      if (fd >= 0) {
        char bytes[128]; // longer than any mark
        struct stat is;
        const bool regular = !fstat(fd, &is) && S_ISREG(is.st_mode);
        ssize_t n = -1;
        while (regular && (n = read(fd, bytes, sizeof bytes)) < 0 && errno == EINTR) {}
        error = errno;
        close(fd);
        if (!regular) return problem = "its mark.txt is not a regular file", WRONG;
        if (n < 0) return problem = std::string("its mark.txt cannot be read: ") + strerror(error), UNREAD;
        const std::string text(bytes, (size_t)n), first = "cp-folder 1\nkind ", head = first + kind + "\ncommitment ";
        const std::string its = text.size() == head.size() + 65 && !text.compare(0, head.size(), head) && text.back() == '\n' ? text.substr(head.size(), 64) : "";
        if (is_commitment_id(its)) return problem.clear(), FITS;
        // Only for the words: a mark of another kind, as the example front end's jobs directory has.
        const size_t end = text.compare(0, first.size(), first) ? std::string::npos : text.find('\n', first.size());
        const std::string its_kind = end == std::string::npos ? std::string() : text.substr(first.size(), end - first.size());
        if (its_kind != kind && fits(its_kind, 32, "abcdefghijklmnopqrstuvwxyz-")) return problem = "its mark.txt is of another kind, " + its_kind + ": the folder serves another purpose", WRONG;
        return problem = "its mark.txt is malformed", WRONG;
      }
      if (error == ELOOP) return problem = "its mark.txt is a link", WRONG;
      if (error != ENOENT)
        return problem = std::string("its mark.txt cannot be opened: ") + strerror(error) +
                         (error == EACCES ? " (what a run under sudo writes belongs to root: run the service and the client both with sudo, or neither)" : ""), UNREAD;
      if (made || !create) return problem = "it has no mark.txt", MISSING;
      std::vector<std::string> names;
      if (!names_in(dir_fd, names, true)) return error = errno, problem = std::string("it has no mark.txt and cannot be listed: ") + strerror(error), UNREAD;
      std::sort(names.begin(), names.end()); // so that of several faults the same one is reported each time
      bool others = false; // a name that neither cp_prove nor a client puts there
      std::string flaw;    // the first fault in in, out or stop
      for (const std::string &name : names) {
        std::string why;
        if (is_mark_name(name) || (!strcmp(kind, "spool") && a_client_puts_first(dir_fd, name, why))) continue;
        if (why.empty()) others = true;
        if (flaw.empty()) flaw = why;
      }
      if (others) return problem = "it is not empty and has no mark.txt", MISSING;
      if (!flaw.empty()) return problem = "it has no mark.txt, and " + flaw, ILL_LAID;
      if (const Mark step = create_mark(dir_fd, mark_text(kind), problem); step != FITS) return step;
      // The next round reads the file back: another program may have written its own first.
    }
  }
  // The beginning of the message for a problem that check_mark() reports.
  static std::string mark_error(const std::string &path, const char *kind, const std::string &problem) {
    return shown(path) + " is not a " + (strcmp(kind, "out") ? "spool directory" : "staging folder of cp_prove -out") + ": " + problem;
  }

  bool took_over = false; // open_one_batch() has succeeded in a directory in which it first removed what an earlier batch had staged
  // Set when open() or open_one_batch() succeeds. If the call fails, the descriptors may be those
  // of a directory that it has rejected, so the calls that clear the staging directory (tidy(),
  // drop_what_is_staged()) touch no file until this is set. After a failed opening,
  // make no other call.
  bool opened = false;
  const char *why = ""; // what went wrong in the last call that failed: constant text, or the text of names_a_part
  std::string names_a_part; // a message of open() or open_one_batch() that names a part, a directory or the mark's problem

private:
  static constexpr const char *LOCK_IS_HELD = "another cp_prove serves this spool or clears it";
  static constexpr const char *EXISTS = "the folder named with -out exists already, and nothing in it was touched. Name a folder that does not exist yet, and remove nothing by hand. If "
                                        "a cp_prove failed or was killed there, also run cp_prove -clear on that folder, so that nothing that it staged stays there: that does "
                                        "not make the folder fit for -out again";
  static constexpr const char *NOT_OPENED = "the folder named with -out, or a folder on the way to it, cannot be opened, and nothing was touched (what a run under sudo makes belongs to "
                                            "root). Name a folder that does not exist yet, below a directory of your own, and remove nothing by hand. If a cp_prove failed or was "
                                            "killed in the folder that you named, also run cp_prove -clear on it, as the user who ran that cp_prove";
  // Whether `name` has 1 to `most` characters, each a digit or one of `others`.
  static bool fits(const std::string &name, size_t most, const char *others) {
    return !name.empty() && name.size() <= most && name.find_first_not_of(std::string("0123456789") + others) == std::string::npos;
  }
  // The types of directory that cp_prove creates: the staging directory of a single batch and that
  // of a spool directory, and a response's.
  enum Kind { STAGING, SPOOL_STAGING, RESPONSE, NO_KIND };
  // Whether cp_prove creates a file of this name in a directory of this kind.
  static bool lays_a_file(Kind kind, const std::string &name) { return kind == RESPONSE && (name == "y" || name == "proof" || name == "failed"); }
  // The kind of the directory of this name that cp_prove creates in a directory of this kind, or
  // NO_KIND if it creates none.
  static Kind lays_a_folder(Kind kind, const std::string &name) {
    if (kind == STAGING && fits(name, 64, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-")) return RESPONSE;
    return kind == SPOOL_STAGING && !id_of(name + ".req").empty() ? RESPONSE : NO_KIND;
  }
  // The body of clear_staged(). folder: an open directory of the kind `kind`; path: its path.
  static void clear_in(int folder, Kind kind, const std::string &path, NotCleared &rest) {
    std::vector<std::string> names;
    // If the listing fails part way, the names that were read are still processed.
    if (!names_in(folder, names, true)) rest.refuse(path, "READ");
    std::sort(names.begin(), names.end()); // so a second call reports what stays in the same order
    for (const std::string &name : names) {
      const std::string full = path + "/" + name;
      if (kind == STAGING && is_mark_name(name)) continue; // a single batch's mark: kept, not reported
      struct stat is;
      if (fstatat(folder, name.c_str(), &is, AT_SYMLINK_NOFOLLOW)) {
        rest.refuse(full, "READ");
        continue;
      }
      if (!S_ISDIR(is.st_mode)) {
        const bool staged_bytes = kind == RESPONSE && S_ISREG(is.st_mode);
        if (!lays_a_file(kind, name))
          rest.foreign.push_back(full);
        else if (staged_bytes && is.st_nlink != 1) // the other name may be someone's own file
          rest.foreign.push_back(full + " (it has a second name)");
        else if (staged_bytes && !overwrite_file(folder, name))
          rest.refuse(full, "OVERWRITTEN"); // it stays, so that a second call can overwrite it
        else if (unlinkat(folder, name.c_str(), 0))
          rest.refuse(full, "REMOVED");
        continue;
      }
      const Kind below = lays_a_folder(kind, name);
      const int inner = below == NO_KIND ? -1 : openat(folder, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (below == NO_KIND)
        rest.foreign.push_back(full);
      else if (inner < 0)
        rest.refuse(full, "READ");
      if (inner < 0) continue;
      const size_t reported = rest.foreign.size() + rest.refused.size();
      clear_in(inner, below, full, rest);
      // If the removal fails because an entry in the directory stayed, that entry is already
      // reported, and the directory is only synced. Any other failure is reported here.
      if (unlinkat(folder, name.c_str(), AT_REMOVEDIR)) {
        if (reported == rest.foreign.size() + rest.refused.size())
          rest.refuse(full, "REMOVED");
        else if (fsync(inner))
          rest.refuse(full, "SYNCED TO THE DISK");
      }
      close(inner);
    }
  }
  struct Answer {
    bool failed = false;
    unsigned char tag[32] = {0}; // set by `tagged`
    ~Answer() { explicit_bzero(tag, sizeof tag); }
  };
  std::string dir, tmp_path;
  static constexpr const char *NOT_NAMED = "a response's file cannot be given its time or its name in out/";
  int T = 0;
  uint32_t V = 0;
  int root = -1, in = -1, work = -1, out = -1, tmp = -1;
  bool plain = false; // true for a single batch (`open_one_batch`)
  static constexpr const char *ENDS[4] = {".y", ".tag", ".proof", ".failed"}; // of a response's files in out/
  bool is_answered(const std::string &id) const {
    for (const char *end : ENDS)
      if (is_there(out, id + end)) return true;
    return false;
  }
  bool withdraw(const std::string &id) const {
    for (const char *end : ENDS) unlinkat(out, (id + end).c_str(), 0);
    return !is_answered(id);
  }
  // take() for a single batch: takes every name that requests_in() returns, in that order. The j-th
  // name gets batch slot j whether or not its file is well formed, because the slot number is part
  // of the labels of a request's private random values. For a malformed file "failed" is staged and
  // its entry has no tokens. If none is well formed, nothing is taken. More than `most` names: -1.
  bool folder_is_taken = false;
  int take_the_folder(std::vector<Request> &taken, int most) {
    if (folder_is_taken) return 0;
    folder_is_taken = true;
    std::vector<std::string> names;
    {
#ifdef __APPLE__
      char path[1024];
      if (fcntl(in, F_GETPATH, path)) path[0] = 0;
#else
      char path[64];
      snprintf(path, sizeof path, "/proc/self/fd/%d", in);
#endif
      std::string odd;
      if (!requests_in(path, names, odd))
        return fault("the folder named with -in cannot be read, or holds something that is no file <name>.req"), -1;
    }
    if ((int)names.size() > most) return fault("the folder named with -in holds more requests than a batch has slots"), -1;
    int well_formed = 0;
    for (const std::string &name : names) {
      Request r;
      r.id = name;
      if (read_request(in, name + ".req", "", r.tokens) == A_REQUEST)
        well_formed++;
      else if (r.tokens.clear(), !stage_failed(name))
        return -1;
      taken.push_back(std::move(r));
    }
    if (!well_formed) taken.clear();
    return (int)taken.size();
  }
  std::mutex guard;                      // protects `waiting` and what is under out/.tmp/
  std::map<std::string, Answer> waiting; // by id: the staged responses
  std::atomic<bool> closed{false};
  std::atomic<bool> left_as_it_was{false}; // set by `drop_what_is_staged`
  mutable std::atomic<bool> staging_is_gone{false}; // set by tidy() once it has removed a single batch's .tmp

  // The syncing thread. A sync on a busy disk can take arbitrarily long, so a response's files are
  // not synced when written or named; only zero_what_is_staged() syncs at once. This thread syncs a
  // released response's files and out/ while the next batch is proved, and only then removes its
  // request from work/ and syncs work/. So after a power loss a request that was taken has a
  // complete response or is in work/.
  std::thread settler;
  std::mutex settle_guard; // protects `named`, `settling` and `settle_ends`
  std::condition_variable settle_news;
  std::vector<std::string> named; // ids of responses that are released and not yet synced
  bool settling = false, settle_ends = false;
  std::atomic<bool> settle_failed{false};
  void settle_for_ever() {
    for (;;) {
      std::vector<std::string> ids;
      {
        std::unique_lock<std::mutex> lock(settle_guard);
        settle_news.wait(lock, [&] { return settle_ends || !named.empty(); });
        if (named.empty()) return;
        ids.swap(named), settling = true;
      }
      if (const int ms = slow_disk_ms) usleep((useconds_t)ms * 1000);
      bool good = true;
      const long long begun = steady_us();
      for (const std::string &id : ids)
        for (const char *end : ENDS) {
          // O_NONBLOCK: without it, opening a FIFO under such a name would block this thread until
          // a writer came, perhaps forever. With it fsync() fails on the FIFO, and the round fails.
          const int fd = openat(out, (id + end).c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
          if (fd < 0) continue; // no such file, or the front end has delivered the response and removed it
          if (fsync(fd)) good = false;
          close(fd);
        }
      const long long files_done = steady_us();
      if (fsync(out)) good = false;
      const long long out_done = steady_us();
      if (good && !plain) {
        for (const std::string &id : ids) unlinkat(work, (id + ".req").c_str(), 0); // the request's file; a directory of that name stays
        // The removals are synced too: otherwise a power loss could bring a request back to work/,
        // and the next open() would withdraw its response, which the front end may have delivered,
        // and answer "failed".
        if (fsync(work)) good = false;
      }
      syncing_us += files_done - begun, syncing_ms = (out_done - files_done) / 1000, removing_ms = (steady_us() - out_done) / 1000;
      {
        std::lock_guard<std::mutex> lock(settle_guard);
        settling = false;
        if (!good) settle_failed = true;
      }
      settle_news.notify_all();
    }
  }

  // Creates <id>.tag in out/, directly under its final name. If the name is taken, the file there
  // stays and the call succeeds, as with a response's other files. both: null, or the file's times.
  bool lay_the_tag(const std::string &id, const Answer &answer, const struct timespec *both) const {
    unsigned char key[32] = {0}, tag[32];
    if (answer.failed && !tag_key(id, key)) return false;
    if (answer.failed)
      tag_of(id, key, nullptr, nullptr, tag);
    else
      memcpy(tag, answer.tag, sizeof tag);
    explicit_bzero(key, sizeof key);
    const int fd = openat(out, (id + ".tag").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    const bool taken = fd < 0 && errno == EEXIST;
    const bool written = fd >= 0 && write(fd, tag, sizeof tag) == (ssize_t)sizeof tag && !(both && futimens(fd, both));
    explicit_bzero(tag, sizeof tag);
    return taken || (fd >= 0 && !close(fd) && written);
  }
  // Does the work of release(). Call it with `guard` held.
  bool name_them(const std::map<std::string, Answer> &answers, long long time_ms) {
    const struct timespec t = {(time_t)(time_ms / 1000), (long)(time_ms % 1000) * 1000000};
    const struct timespec both[2] = {t, t};
    for (const auto &[id, answer] : answers) {
      const int folder = openat(tmp, id.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (folder < 0) return fault("a response's folder cannot be opened");
      bool good = true;
      for (const char *file : {"y", "tag", "failed", "proof"}) { // y, tag, proof; or tag, failed
        if (!good) break; // after a step that failed nothing more is named, so the response stays incomplete
        if (!strcmp(file, "tag")) {
          if (!plain && !lay_the_tag(id, answer, time_ms >= 0 ? both : nullptr)) good = false;
          continue;
        }
        if (answer.failed != !strcmp(file, "failed")) continue;
        if (time_ms >= 0 && utimensat(folder, file, both, 0)) {
          good = false;
          break;
        }
        const std::string name = id + "." + file;
        // This call replaces nothing. Where it is not supported: renameat(), if the name is free.
        int named = (int)syscall(SYS_renameat2, folder, file, out, name.c_str(), 1 /* RENAME_NOREPLACE */);
        if (named && (errno == EINVAL || errno == ENOSYS))
          named = is_there(out, name) ? (errno = EEXIST, -1) : renameat(folder, file, out, name.c_str());
        if (named && errno != EEXIST) good = false;
      }
      close(folder);
      if (!good) return fault(NOT_NAMED);
      remove_folder(tmp, id);
    }
    // Nothing is synced here. The syncing thread (settle_for_ever()) syncs the responses to disk,
    // and only then do their requests leave work/.
    {
      std::lock_guard<std::mutex> lock(settle_guard);
      for (const auto &[id, answer] : answers) named.push_back(id);
    }
    settle_news.notify_all();
    return true;
  }

  static void wipe(std::string &bytes) {
    if (!bytes.empty()) explicit_bzero(&bytes[0], bytes.size());
  }
  // False if the file is not a regular file, cannot be read, or changes its length during the read.
  static bool read_whole(int folder, const char *name, std::string &bytes) {
    const int fd = openat(folder, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    struct stat s;
    bool whole = fd >= 0 && !fstat(fd, &s) && S_ISREG(s.st_mode);
    if (whole) bytes.resize((size_t)s.st_size);
    for (size_t at = 0; whole && at < bytes.size();) {
      const ssize_t n = pread(fd, &bytes[at], bytes.size() - at, (off_t)at);
      if (n < 0 && errno == EINTR) continue;
      whole = n > 0;
      at += whole ? (size_t)n : 0;
    }
    char more;
    ssize_t beyond = 0;
    while (whole && (beyond = pread(fd, &more, 1, (off_t)bytes.size())) < 0 && errno == EINTR) {
    }
    whole = whole && !beyond;
    if (fd >= 0) close(fd);
    return whole;
  }
  bool read_staged(const std::string &id, std::string &y, std::string &proof) const {
    const int folder = openat(tmp, id.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    const bool read = folder >= 0 && read_whole(folder, "y", y) && read_whole(folder, "proof", proof);
    if (folder >= 0) close(folder);
    return read;
  }
  // The two lengths in bytes, each as a 64-bit little-endian number.
  static void lengths_of(const std::string &y, const std::string &proof, unsigned char lengths[16]) {
    for (int i = 0; i < 8; i++)
      lengths[i] = (unsigned char)((uint64_t)y.size() >> 8 * i), lengths[8 + i] = (unsigned char)((uint64_t)proof.size() >> 8 * i);
  }
  // Recomputes the key of the response's tag from the request's file in work/. The caller must zero
  // the key after use. False if work/ has no file <id>.req whose bytes hash to the id.
  bool tag_key(const std::string &id, unsigned char key[32]) const {
    std::vector<uint32_t> words;
    const bool found = read_request(work, id + ".req", id, words, key) != NOT_ITS_NAME;
    if (!words.empty()) explicit_bzero(words.data(), words.size() * 4);
    return found;
  }
  // Reads a staged response's files and computes its tag from these bytes. False if either step fails.
  bool tagged(const std::string &id, Answer &answer) {
    std::string y, proof;
    unsigned char key[32] = {0};
    const bool read = read_staged(id, y, proof) && tag_key(id, key);
    if (read) tag_of(id, key, &y, &proof, answer.tag);
    wipe(y), wipe(proof);
    explicit_bzero(key, sizeof key);
    return read;
  }
  // Zeroes and syncs a staged response's files; writes through no symbolic or hard link. The file
  // system or disk may keep old blocks. False unless all were zeroed and synced. Call it with
  // `guard` held.
  bool zero_what_is_staged(const std::string &id) const {
    const int folder = openat(tmp, id.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (folder < 0) return errno == ENOENT; // nothing is staged under the id, so nothing is left unchanged
    std::vector<std::string> files;
    bool zeroed = names_in(folder, files);
    for (const std::string &file : files) zeroed = overwrite_file(folder, file) && zeroed;
    close(folder);
    return zeroed;
  }
  // The same for one file in an open directory. False, with errno set, unless all of it was zeroed
  // and synced: EACCES if it is not a regular file of this user's, EMLINK if it has a second name.
  static bool overwrite_file(int folder, const std::string &file) {
    static const char zeros[1 << 16] = {0}; // no allocation: memory may be what ran out
    const int fd = openat(folder, file.c_str(), O_WRONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat s;
    bool own = !fstat(fd, &s);
    if (own && !(S_ISREG(s.st_mode) && s.st_uid == geteuid())) own = false, errno = EACCES;
    if (own && s.st_nlink != 1) own = false, errno = EMLINK;
    off_t left = own ? s.st_size : -1;
    while (left > 0) {
      const ssize_t n = write(fd, zeros, (size_t)std::min<off_t>(left, (off_t)sizeof zeros));
      if (n < 0 && errno == EINTR) continue;
      if (!n) errno = EIO;
      if (n <= 0) break;
      left -= n;
    }
    const bool zeroed = !left && !fdatasync(fd);
    const int error = errno;
    close(fd);
    errno = error;
    return zeroed;
  }
  // Overwrites and removes a staged response; what it cannot overwrite it removes all the same.
  // True if both succeed. Call it with `guard` held.
  bool erase(const std::string &id) const {
    const bool zeroed = zero_what_is_staged(id);
    return remove_folder(tmp, id) && zeroed;
  }
  // Overwrites a staged response's files, removes them and stages "failed" instead. If a file
  // cannot be overwritten it still does the rest, and returns false. Call it with `guard` held.
  bool turn_to_failed(const std::string &id, Answer &answer) {
    if (answer.failed) return true;
    answer.failed = true;
    explicit_bzero(answer.tag, sizeof answer.tag);
    const bool zeroed = zero_what_is_staged(id);
    if (!remove_folder(tmp, id) || mkdirat(tmp, id.c_str(), 0700)) return fault("a response's folder cannot be created");
    return write_files(id, {{"failed", "", 0}}) && (zeroed || fault("a withheld response's file cannot be overwritten"));
  }

  bool fault(const char *what) {
    why = what;
    return false;
  }
  // "<id>.req" -> "<id>", or "" if the name is not a request's. An id is 64 lower-case hex digits.
  static std::string id_of(const std::string &name) {
    if (name.size() != 68 || name.compare(64, 4, ".req")) return "";
    const std::string id = name.substr(0, 64);
    for (unsigned char c : id)
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return "";
    return id;
  }
  static bool is_there(int folder, const std::string &name) {
    struct stat s;
    return !fstatat(folder, name.c_str(), &s, AT_SYMLINK_NOFOLLOW);
  }
  // True if the entry named stop in the open directory `top` is an empty regular file, not a link.
  static bool is_a_stop(int top) {
    struct stat s;
    return !fstatat(top, "stop", &s, AT_SYMLINK_NOFOLLOW) && S_ISREG(s.st_mode) && !s.st_size;
  }
  // True if the open directory belongs to this process's user, or to the account `or_of` unless
  // that is 0, and gives group and others no permission. The system's own permission checks do not
  // stop root, so only this comparison keeps a process that runs as root out of another user's
  // directory.
  static bool is_closed_to_others(int folder, uid_t or_of = 0) {
    struct stat s;
    return !fstat(folder, &s) && (s.st_uid == geteuid() || (or_of && s.st_uid == or_of)) && !(s.st_mode & 077);
  }
  // For open(): creates the directory `name` in the open directory `parent`, with mode 0700, unless
  // the name exists, and opens it without following a link. Returns its descriptor, or -1, with
  // `why` set, if a step fails or is_closed_to_others() is false. `below`: its path from the spool
  // directory, such as "/out/.tmp", for not_the_users() and cannot_be_opened().
  int subfolder(int parent, const char *name, const char *below) {
    if (mkdirat(parent, name, 0700) && errno != EEXIST) return fault("a subfolder cannot be created"), -1;
    const int fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return cannot_be_opened(errno, below), -1;
    if (is_closed_to_others(fd)) return fd;
    not_the_users(fd, below);
    close(fd);
    return -1;
  }
  // Sets `why` for the spool directory (`below` is "") or one of its subdirectories when
  // is_closed_to_others() is false: the message names the directory, the owner and mode found on
  // it, and what to do. `folder`: its open descriptor. Returns false.
  bool not_the_users(int folder, const char *below) {
    struct stat s {};
    char found[64] = "its owner and mode cannot be read";
    if (!fstat(folder, &s)) snprintf(found, sizeof found, "owner %lu, mode %04o", (unsigned long)s.st_uid, (unsigned)(s.st_mode & 07777));
    const bool only_the_mode = (s.st_uid == geteuid() || (!*below && typed_sudo && s.st_uid == typed_sudo)) && (s.st_mode & 077);
    names_a_part = shown(dir + below) + " (" + found + ") belongs to another user, or is open to its group or to others. The spool directory and each folder in it must belong to the user who runs the service (you are " +
                   std::to_string((unsigned long)geteuid()) + "), with mode 0700; under sudo the spool directory itself may also be the account's that typed sudo. " +
                   (only_the_mode ? std::string("Its owner is right and only its mode is open: find out who could write there; then ") +
                                        (geteuid() ? "close it (chmod 700)" : "name a new directory, made with mode 0700") // (root is never told to chmod: see `open_by_parts`)
                                  : "What a run under sudo makes belongs to root: run the service and the client both with sudo, or neither; else name a spool directory of your own");
    return fault(names_a_part.c_str());
  }
  // The names in a directory, without "." and ".." and, unless `all` is set, ".tmp".
  // False, with errno set, if it cannot be read.
  static bool names_in(int folder, std::vector<std::string> &names, bool all = false) {
    names.clear();
    const int again = openat(folder, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC); // closedir() closes its descriptor
    DIR *d = again < 0 ? nullptr : fdopendir(again);
    if (!d) {
      if (again >= 0) close(again);
      return false;
    }
    const struct dirent *e;
    for (errno = 0; (e = readdir(d)); errno = 0) // push_back() can set errno too
      if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..") && (all || strcmp(e->d_name, ".tmp"))) names.push_back(e->d_name);
    const int error = errno;
    closedir(d);
    errno = error;
    return !error;
  }
  // Removes a directory that contains only files, or a file of that name. True if nothing is left.
  static bool remove_folder(int parent, const std::string &name) {
    const int folder = openat(parent, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (folder >= 0) {
      std::vector<std::string> files;
      names_in(folder, files);
      for (const std::string &file : files) unlinkat(folder, file.c_str(), 0);
      close(folder);
      unlinkat(parent, name.c_str(), AT_REMOVEDIR);
    } else {
      unlinkat(parent, name.c_str(), 0);
    }
    return !is_there(parent, name);
  }
  // For check_mark(): whether `name`, at the top of the open directory `top`, is what a client may
  // put there before the first start (driver/spool.py): in/ with only the directory .tmp and regular
  // files <64 hex digits>.req; an empty out/; an empty file stop, which open() then removes. The
  // directories are this user's and closed to others, and nothing is a link. What is inside in/.tmp
  // is not looked at. why: receives the fault in an entry of one of the three names. Limit: nothing
  // says for which commitment waiting requests were submitted: a first start for another serves them.
  static bool a_client_puts_first(int top, const std::string &name, std::string &why) {
    if (name == "stop") return is_a_stop(top) || (why = "stop is not an empty file", false);
    if (name != "in" && name != "out") return false;
    const int folder = openat(top, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    std::vector<std::string> entries;
    if (folder < 0 || !is_closed_to_others(folder) || !names_in(folder, entries, true))
      why = name + "/ is not a folder of yours with mode 0700 (if a client has made it: run the service and the client both with sudo, or neither)";
    else if (name == "out" && !entries.empty())
      why = "out/ is not empty";
    std::sort(entries.begin(), entries.end());
    for (const std::string &entry : entries) {
      struct stat is;
      if (!why.empty()) break;
      const bool seen = !fstatat(folder, entry.c_str(), &is, AT_SYMLINK_NOFOLLOW);
      if (entry == ".tmp" && !(seen && S_ISDIR(is.st_mode) && is.st_uid == geteuid() && !(is.st_mode & 077))) why = "in/.tmp is not a folder of yours with mode 0700";
      if (entry != ".tmp" && !(seen && S_ISREG(is.st_mode) && !id_of(entry).empty())) why = "in/ holds " + shown(entry) + ", which is no request: a request is a regular file named <64 hex digits>.req";
    }
    if (folder >= 0) close(folder);
    return why.empty();
  }
  // For a refusal: the names at the top of the open directory `top`, at most 20, without unfinished
  // marks.
  static std::string at_the_top(int top) {
    std::vector<std::string> names;
    names_in(top, names, true);
    names.erase(std::remove_if(names.begin(), names.end(), is_mark_name), names.end());
    std::sort(names.begin(), names.end());
    std::string all;
    for (size_t k = 0; k < names.size() && k < 20; k++) all += (k ? ", " : "") + shown(names[k]);
    return names.size() > 20 ? all + " and " + std::to_string(names.size() - 20) + " more" : all;
  }
  // Writes mark.txt as lay() writes a request: the whole text under a temporary name, synced, then
  // renamed by a call that replaces nothing. To find the file already there is no error. Returns
  // FITS, or NOT_MADE or NOT_SYNCED with `problem` set.
  static Mark create_mark(int dir_fd, const std::string &text, std::string &problem) {
    const std::string tmp_name = MARK_TMP_PREFIX + std::to_string((long long)getpid()) + "." + std::to_string(steady_us());
    auto failed = [&](const char *step, int error, bool created) {
      if (created) unlinkat(dir_fd, tmp_name.c_str(), 0);
      return problem = std::string("it has no mark.txt yet, and the file cannot be ") + step + ": " + strerror(error), NOT_MADE;
    };
    const int fd = openat(dir_fd, tmp_name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return failed("created", errno, false);
    const ssize_t n = write(fd, text.data(), text.size());
    int error = n < 0 ? errno : n != (ssize_t)text.size() ? EIO : fsync(fd) ? errno : 0; // a short write sets no errno
    if (close(fd) && !error) error = errno;
    if (error) return failed("written", error, true);
    int status = (int)syscall(SYS_renameat2, dir_fd, tmp_name.c_str(), dir_fd, MARK_FILE, 1 /* RENAME_NOREPLACE */); // 0: the file has its name
    if (status && (errno == EINVAL || errno == ENOSYS)) { // no such call on this file system: linkat(), then unlinkat()
      status = linkat(dir_fd, tmp_name.c_str(), dir_fd, MARK_FILE, 0);
      if (!status) unlinkat(dir_fd, tmp_name.c_str(), 0);
    }
    if (status && errno != EEXIST) return failed("put in place", errno, true);
    if (status) unlinkat(dir_fd, tmp_name.c_str(), 0); // another program's mark.txt was there first
    if (fsync(dir_fd)) return error = errno, problem = std::string("its mark.txt is there, but the folder cannot be synced to the disk: ") + strerror(error), NOT_SYNCED;
    return FITS;
  }
  // Returns the id's entry in `waiting`. If there is none, creates the response's directory and a
  // new entry. Returns null, with `why` set, on failure. Call it with `guard` held.
  Answer *note(const std::string &id) {
    if (closed) return fault("the process is ending"), nullptr;
    const auto found = waiting.find(id);
    if (found != waiting.end()) return &found->second;
    const long long begun = steady_us();
    // An existing name is not this process's: open() has emptied out/.tmp/, and take() claims no
    // id that has one. Anyone who could write there may have put a link or directory under the id.
    const bool made = !mkdirat(tmp, id.c_str(), 0700);
    making_us += steady_us() - begun;
    if (!made) return fault("a response's folder cannot be created"), nullptr;
    return &waiting[id];
  }
  bool write_files(const std::string &id, const std::vector<File> &files) {
    const long long entered = steady_us();
    const int folder = openat(tmp, id.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    making_us += steady_us() - entered;
    // No part of a response is written into a directory or a file that belongs to another user or
    // that group or others can access, or into a file with a second hard link.
    struct stat is;
    if (folder < 0 || fstat(folder, &is) || is.st_uid != geteuid() || (is.st_mode & 077)) {
      if (folder >= 0) close(folder);
      return fault("a response's folder cannot be opened");
    }
    bool good = true;
    for (const File &f : files) {
      const long long making = steady_us();
      // O_NONBLOCK: if the name is a FIFO that no one reads, the open fails instead of blocking.
      const int fd = openat(folder, f.name, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0600);
      const bool own = fd >= 0 && !fstat(fd, &is) && S_ISREG(is.st_mode) && is.st_uid == geteuid() && !(is.st_mode & 077) && is.st_nlink == 1;
      if (!own) good = false;
      const char *p = (const char *)f.bytes;
      size_t left = f.n;
      const long long begun = steady_us();
      making_us += begun - making;
      while (own && left) {
        const ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        p += n, left -= (size_t)n;
      }
      const bool shut = fd >= 0 && !close(fd);
      if (left || !shut) good = false;
      writing_us += steady_us() - begun;
    }
    close(folder); // nothing is synced here, within the batch's time: settle_for_ever() does it
    return good || fault("a response's file cannot be written");
  }

  // Reads <folder>/<name>. NOT_ITS_NAME: it is not a regular file of a request's length whose bytes
  // hash to `id` (single batch: the length is without the nonce, and no hash is compared), or the
  // open, fstat or full read fails. A_TOKEN_OUTSIDE: it is such a file, but a token is not below V
  // or r < e <= T does not hold. Only with A_REQUEST does `tokens` receive the words (words_of()).
  // tag_key: null, or 32 bytes for the tag's key, set unless NOT_ITS_NAME; not for a single batch.
  enum Read { A_REQUEST, A_TOKEN_OUTSIDE, NOT_ITS_NAME };
  Read read_request(int folder, const std::string &name, const std::string &id, std::vector<uint32_t> &tokens,
                    unsigned char *tag_key = nullptr) const {
    const size_t length = plain ? words_of(T) * 4 : bytes_of(T);
    const int fd = openat(folder, name.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    struct stat s;
    std::vector<unsigned char> bytes(length);
    const bool whole = fd >= 0 && !fstat(fd, &s) && S_ISREG(s.st_mode) && s.st_size == (off_t)length &&
                       pread(fd, bytes.data(), length, 0) == (ssize_t)length;
    if (fd >= 0) close(fd);
    Read read = whole && (plain || id_of_bytes(bytes.data(), length) == id) ? A_REQUEST : NOT_ITS_NAME;
    tokens.clear();
    if (read == A_REQUEST && tag_key && !plain) tag_key_of(bytes.data() + words_of(T) * 4, tag_key);
    if (read == A_REQUEST) {
      tokens.resize(words_of(T));
      memcpy(tokens.data(), bytes.data(), words_of(T) * 4);
      uint32_t outside = 0; // every word is examined, wherever the first bad one is
      for (int t = 0; t < T; t++) outside |= tokens[t] >= V;
      outside |= (uint32_t)(tokens[T] >= tokens[T + 1]) | (uint32_t)(tokens[T + 1] > (uint32_t)T);
      if (outside) explicit_bzero(tokens.data(), tokens.size() * 4), tokens.clear(), read = A_TOKEN_OUTSIDE;
    }
    explicit_bzero(bytes.data(), length);
    return read;
  }

  // For subfolder(): sets `why` for a subdirectory that cannot be opened: which, why, for EACCES
  // whose it is and would have to be, and what the caller can do. error: the errno of the open;
  // `below`: as for subfolder(). EACCES is what an account meets at a subdirectory that a run as
  // root or under sudo has made. No message asks anyone to change a directory as root, and no
  // command in one names a path: a printed path is resolved again when it is typed, and may by
  // then lead elsewhere. Returns false.
  bool cannot_be_opened(int error, const char *below) {
    const std::string path = shown(dir + below), untouched = "Nothing that was in the spool directory was touched. ",
                      another = "name a new, empty folder of your own (mkdir -m 700): the requests in this one stay there and get no response";
    if (error == ELOOP || error == ENOTDIR)
      names_a_part = path + " is a link or a file: cp_prove needs a folder of that name and does not follow links. " + untouched +
                     "Look at it (ls -ld). If another folder is the service's spool directory, name that one. If not, " + another;
    else if (error != EACCES)
      names_a_part = path + " cannot be opened: " + strerror(error) + ". " + untouched + "Fix the cause and start again on the same folder";
    else
      names_a_part = path + " cannot be opened: " + strerror(error) + ". " + whose(dir + below) + untouched + // (root is never told to chmod: see `open_by_parts`)
                     (geteuid() ? "If the folder is yours and only its mode is closed, give yourself its permissions back (chmod 700) and start again. If not, you" : "You") + " can " + another +
                     ". Or run every step on this one the same way, with sudo or without: what a run as root or under sudo makes belongs to root";
    return fault(names_a_part.c_str());
  }
  // For a message on the directory `path`, which the caller may not open: whose it is and would
  // have to be. Only for the words: it looks by path and does not follow a link at its last part.
  static std::string whose(const std::string &path) {
    struct stat s;
    char found[80] = "Its owner and mode cannot be read either (a folder above it may be closed)";
    if (!lstat(path.c_str(), &s)) snprintf(found, sizeof found, "It belongs to user %lu%s, with mode %04o", (unsigned long)s.st_uid, s.st_uid ? "" : " (root)", (unsigned)(s.st_mode & 07777));
    return std::string(found) + "; it would have to belong to user " + std::to_string((unsigned long)geteuid()) + ", who runs this, with mode 0700. ";
  }
};
