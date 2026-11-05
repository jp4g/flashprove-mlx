#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds the verifier, cp_verify, from verifier/src/cp_verify.cpp. Needs g++ 13 or later, GNU
# coreutils and findutils, about two minutes and 1.6 GB of memory; no GPU.
# The program goes into bin/ at the checkout's top, or into the directory given after "-"
# (`<output folder>` in the usage text; a relative path starts at the caller's working directory).
# A cp_verify that is already there is overwritten, with a notice on stderr; a build that fails or
# is interrupted leaves it unchanged. The script builds nothing in a given directory that another
# account could tamper with (open_above()), whose parent is missing or whose path has a newline.
# It compiles in a work directory of its own, cp_verify.building.XXXXXX in the output directory,
# which is also the compiler's TMPDIR. It removes only that work directory, and itself writes
# nothing outside the output directory. The compiler gets the caller's environment without seven
# variables that can name a file for it to append to (see "; unset" below); the rest of that
# environment, PATH included, can still tell it to write elsewhere. What can stay besides
# cp_verify: an output directory that the script created, even if nothing is built; and the work
# directory, if the compiler left a temporary file in it, if the script's check right after
# creating it fails (it cannot be examined, is not empty or is no longer the same directory), or
# if the script is killed before finish() runs, as by SIGKILL. No later run removes or needs it.
# The script sends no signal: the compiler runs on if the script alone is killed, and a SIGTERM or
# SIGHUP sent to the script alone while it compiles takes effect only when the compiler has ended.
#   --field  accepts only 2 and changes nothing: there is one field, p = 2^64 - 2^32 + 1, number 2.
#   --clang  compiles with clang++ (only 18.1.3 has been tried), for comparison with the g++ build
#            (verifier/tests/clang_build_test.sh). The output directory must then be given, so that
#            this build does not replace the g++ build in the default directory by accident.
#            Figures and test runs in this repository use the g++ build unless they say otherwise.
# usage: build.sh [--field 2] [- <output folder>]     or: build.sh [--field 2] --clang - <output folder>
set -euo pipefail
unset CDPATH # with CDPATH set, a cd below could end in a directory of the same name elsewhere
here=$(cd "$(dirname "$0")" && pwd -P) || exit 2 # without links, as the places that open_above() names
usage() {
  echo "usage: build.sh [--field 2] [- <output folder>]     or: build.sh [--field 2] --clang - <output folder>" >&2
  exit 2
}
dw=8
field=2
if [[ ${1:-} == --field ]]; then
  [[ $# -ge 2 ]] || usage
  if [[ $2 != 2 ]]; then
    echo "build.sh: there is one field, number 2 (p = 2^64 - 2^32 + 1): --field $(printf '%q' "$2") cannot be built." >&2
    exit 2
  fi
  field=$2
  shift 2
fi
clang=0
if [[ ${1:-} == --clang ]]; then
  clang=1
  shift
fi
[[ ($# -eq 0 && $clang == 0) || ($# -eq 2 && $1 == - && -n $2) ]] || usage
named=${2:-} # empty: the default output directory, inside the checkout
out=${2:-"${here%/*}/bin"}
# The program built in the output directory will be run. So this script builds only where no other
# account can put a file into that directory, or another directory in its place. open_above(), which
# checks the path, is the same in every script of this repository that has it.
# The default directory, inside the checkout, has one exception. If a place (a directory or link) at
# or above the checkout's top is open to another account, the script prints a note and builds:
# whoever can change that place can change this script and its sources too, so rejecting the
# directory (refuse()) would protect nothing. The script then rejects nothing about the directory
# itself: it skips the check of the mode, and what the search of the directory finds is only a note.

# <absolute path of a directory>: walks it from the top as the system does, through each symbolic
# link. Returns 0 at the first place where another account may be able to swap what lies below: one
# whose owner is not this account, root or, under sudo, the account that ran sudo; a directory that
# its group or others can write in without the sticky bit; or one that is missing, is no directory
# or cannot be examined. It then sets place, why and cure for the message. Else it returns 1 and
# sets real, the path without links that it has looked at: the path to use. Keep it in step with
# others_can_change() in run_suite.py.
open_above() {
  local at=/ here=/ part target owner mode kind me links=0 parts=()
  place='' why='' cure='' real=''
  # A here-string needs a writable TMPDIR before bash 5.1. Up to a NUL: names can have newlines.
  # id, not $EUID: bash takes a value for that name from the caller's environment.
  if ! me=$(id -u 2>/dev/null) || [[ ${1:-} != /* ]] ||
    ! IFS=/ read -r -d '' -a parts < <(printf '%s\0' "$1"); then
    printf -v place %q "${1:-}"
    why='it could not be examined'
    return 0
  fi
  set -- ${parts[@]+"${parts[@]}"}
  while :; do
    if ! IFS=' ' read -r owner mode kind < <(
      LC_ALL=C stat -c '%u %a %F' -- "$here" 2>/dev/null
    ); then
      why='it could not be examined'
      if [ ! -e "$here" ] && [ ! -L "$here" ] && [ -x "$at" ]; then why='it does not exist'; fi
    elif [ "$owner" != 0 ] && [ "$owner" != "$me" ] &&
      { [ "$me" != 0 ] || [ "$owner" != "${SUDO_UID:-}" ]; }; then
      why='owned by another account'
      [ "$kind" != 'symbolic link' ] || why='a symbolic link that belongs to another account'
    elif [ "$kind" = directory ]; then
      if ((8#$mode & 8#22)) && ! ((8#$mode & 8#1000)); then
        why='its group or others can write in it, and it has no sticky bit'
        [ "$owner" != "$me" ] || [ "$me" = 0 ] || cure=yes # root's may be shared on purpose
      fi
    elif [ "$kind" != 'symbolic link' ]; then
      why='it is not a directory'
    else # the link's parts come before the rest of the path
      links=$((links + 1))
      if [ "$links" -gt 40 ] ||
        ! IFS= read -r -d '' target < <(readlink -z -- "$here" 2>/dev/null) ||
        ! IFS=/ read -r -d '' -a parts < <(printf '%s\0' "$target"); then
        why='it could not be examined'
      fi
    fi
    if [ -n "$why" ]; then
      printf -v place %q "$here"
      cure=${cure:+"; if $place is yours, close it with chmod go-w"}
      return 0
    fi
    if [ "$kind" = directory ]; then
      at=$here
    else
      set -- ${parts[@]+"${parts[@]}"} "$@"
      [[ $target != /* ]] || at=/
    fi
    while :; do # the next name; ".." as the system takes it, from the directory reached
      if ! (($#)); then
        real=$at
        return 1
      fi
      part=$1
      shift
      case $part in
        '' | .) ;;
        ..) at=${at%/*} at=${at:-/} ;;
        *) break ;;
      esac
    done
    here=${at%/}/$part
  done
}
# Prints what open_above() has found and exits with status 2. One exception: for the default output
# directory, if the place is at or above the checkout's top and the fault is its owner or its mode,
# it prints a note, sets `excused` and returns.
refuse() {
  if [[ -z $named && -z $itself && $why != it\ * ]]; then
    excused=1
    echo "build.sh: note: another account may be able to change $place ($why), under which this checkout lies, and so could" \
      "change what is built, and this script itself; for a build that no other account can touch, keep the checkout below" \
      "directories that only you can write in$cure" >&2
    return
  fi
  local how='name a directory of your own below directories that only you or root can change (such as one in your home'
  how+=' directory): end the call with "- DIRECTORY"'
  if [ "$why" = 'it does not exist' ]; then
    echo "build.sh: nothing was built: $place, above the output directory, does not exist: make it first; build.sh makes" \
      "only the last part of the path" >&2
  elif [[ $why == it\ * ]]; then # it is not a directory; it could not be examined
    echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") cannot be used: $place: $why: $how" >&2
  else
    echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") is, or lies below, $place, which another" \
      "account may be able to change ($why): it could put another directory in its place while the build runs and choose" \
      "what you later run as cp_verify: $how$cure" >&2
  fi
  exit 2
}
# itself: set if the place found is the output directory itself or, for the default, is not at or
# above the checkout's top.
itself='' excused='' made='' # made: set once this call has created the output directory
[[ $out == /* ]] || out=$PWD/$out
while [[ $out == */ && $out != / ]]; do out=${out%/}; done
if [[ $out == *$'\n'* ]]; then # the last line prints the path
  echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") has a newline in its path: name another directory" >&2
  exit 2
fi
# An output directory that does not exist is created, not writable by its group or by others: only
# its last part, by a plain mkdir, which fails if anything has taken the name. The parent of a
# directory that the caller names must pass open_above() first.
if [ ! -e "$out" ] && [ ! -L "$out" ]; then
  if [ -n "$named" ]; then
    if open_above "${out%/*}/"; then refuse; fi
    out=${real%/}/${out##*/}
  fi
  if ! (umask go-w && mkdir -- "$out"); then
    echo "build.sh: the output directory $(printf '%q' "$out") cannot be created: see the message above. Name another" \
      "directory." >&2
    exit 2
  fi
  made=1
fi
if open_above "$out/"; then
  # The hint to run chmod (`cure`) is dropped for the output directory itself: what another account
  # put there while it was open would still be there.
  if [ "$place" = "$(printf %q "$(realpath -- "$out" 2>/dev/null)")" ] || [ "$place" = "$(printf %q "$out")" ]; then cure='' itself=1; fi
  # For the default, only a place at or above the checkout's top can be excused, not one that a
  # symbolic link at the default's path leads to.
  [[ -n $named || $(printf %q "${here%/*}")/ == "${place%/}"/* ]] || itself=1
  refuse
  real=$(cd -P -- "$out" && pwd) || exit 2 # the default, behind the note
fi
out=$real # absolute, without links: the path that the walk has looked at (behind a note: as cd -P resolves it)
if [[ $out == *$'\n'* ]]; then # the last line prints the path
  echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") has a newline in its path: name another directory" >&2
  exit 2
fi
# The sticky bit excuses a directory above, not the output directory itself: there any account could still add files beside
# the program.
mode=$(LC_ALL=C stat -c %a -- "$out") || mode=7777 # if stat fails, the directory is refused
if [ -z "$excused" ] && ((8#$mode & 8#1022)); then
  echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") can be written in by its group or by" \
    "others, or has the sticky bit, or could not be examined: name a new directory of your own: end the call with \"- DIRECTORY\"" >&2
  exit 2
fi
# The directory is searched for anything that another account owns, put there while the directory
# was open, and for subdirectories that their group or others can write in, where another account
# can still add files.
# Two owners are trusted, as in open_above(): root, and this account (for root under sudo, the
# account that ran sudo). Every other owner is matched by "greater than" and "less than", never by
# "not": if a file vanishes while find runs (a second build in the same directory creates and
# removes files), every test on its owner is false, and a "not" would turn that into a match.
upper=$(id -u) || upper=0
if [ "$upper" = 0 ] && [[ ${SUDO_UID:-} =~ ^[1-9][0-9]*$ ]]; then upper=$SUDO_UID; fi
# find fails if another build's work directory vanishes while it runs (-ignore_readdir_race covers
# only files), so the search is tried up to three times.
found='' searched=''
for _ in 1 2 3; do
  if [ -n "$searched" ]; then break; fi
  if found=$(find "$out" -ignore_readdir_race -mindepth 1 \( -uid +0 -uid "-$upper" -o -uid "+$upper" -o -type d -perm /022 \) -print -quit 2>/dev/null); then
    searched=yes
  elif [ -n "$found" ]; then
    break
  fi
done
if [[ -z $searched || -n $found ]]; then
  name=${found:+ ($(printf '%q' "$found"))}
  found="the output directory $(printf '%q' "$out") holds something that belongs to another account, or a directory that"
  found+=" its group or others can write in$name, or could not be searched"
  if [ -z "$excused" ]; then
    echo "build.sh: nothing was built: $found: name a new directory: end the call with \"- DIRECTORY\"" >&2
    exit 2
  fi
  echo "build.sh: note: $found" >&2 # the default, below an open place: a note, and the build goes on
fi
umask go-w # nothing that the build makes there can be written by its group or by others, whatever the caller's umask
where=$(printf '%q' "$out") # shell-quoted for the messages, so that any name prints on one line
# roots_bin: where the default output directory belongs to root in a checkout of another account's,
# as after a build under sudo, it prints what that means and the ways on that need no root.
# Otherwise it prints nothing, also where the program's source is not in `here`: that directory
# comes from the name by which the script was called, and through a link elsewhere, or from standard
# input, it is not the script's own.
roots_bin() {
  local top=${here%/*} owner
  [[ -z $named && -n $top && $out == "$top/bin" && -f $here/src/cp_verify.cpp ]] || return 0
  [[ $(stat -c %u -- "$out" 2>/dev/null) == 0 ]] && owner=$(stat -c %u -- "$top" 2>/dev/null) || return 0
  [[ $owner =~ ^[1-9][0-9]*$ ]] || return 0
  printf '%s belongs to root, in %q, which belongs to the account with uid %s. A build that does not run as root' "$where" "$top" "$owner"
  printf ' cannot write into it. To go on without root, name another output directory (end the call with "- DIRECTORY").'
  printf ' The tests and other scripts of the repository look for cp_verify in %s: for them, start again in a new copy of' "$where"
  printf " the repository. That the directory is root's does not protect it: whoever owns %q can replace it, cp_verify included." "$top"
}
# Root has just made the default directory: one note, if it is in a checkout of another account's.
if [[ -n $made && $(id -u) == 0 ]] && told=$(roots_bin) && [[ -n $told ]]; then
  (printf 'build.sh: note: %s\n' "$told" >&2) || true # in a subshell: if nobody reads standard error any more, the build goes on
fi
# The compiler writes into a new directory inside the output directory ($work), and a rename then installs the program.
# So a build that fails or is interrupted leaves an earlier cp_verify unchanged, no unfinished file is ever at the
# program's path, and a symbolic link at that path is replaced, not followed.
work=$(mktemp -d -- "$out/cp_verify.building.XXXXXX") || {
  told=''
  if [[ $(id -u) != 0 ]]; then told=$(roots_bin) || told=''; fi
  if [[ -n $told ]]; then
    printf 'build.sh: nothing was built (see the message above): %s\n' "$told" >&2
  else
    echo "build.sh: nothing was built: no work directory can be created in $where: see the message above. If you have" \
      "no permission to write there, name another output directory: end the call with \"- DIRECTORY\"" >&2
  fi
  exit 2
}
work_id=$(stat -c %d:%i -- "$work") # device and inode
export TMPDIR=$work; unset DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES CC_PRINT_OPTIONS_FILE CC_PRINT_HEADERS_FILE CC_LOG_DIAGNOSTICS_FILE CC_PRINT_PROC_STAT_FILE CC_PRINT_INTERNAL_STAT_FILE # each can name a file to append to
# in_work <command>: runs the command inside the work directory, after checking that this is still the directory created
# above. So nothing is moved out of, or removed from, another directory that has since taken that path.
in_work() (cd -- "$work" && [[ $(stat -c %d:%i ./) == "$work_id" ]] && "$@")
# The work directory must be empty, so that whatever finish() removes there is this build's own.
if ! found=$(in_work ls -Aq) || [[ -n $found ]]; then
  echo "build.sh: nothing was built: the new work directory $(printf '%q' "$work") is not empty, or it is no longer" \
    "the directory that this build created. It was left as it is. Build again." >&2
  exit 2
fi
# Runs at every exit once the traps below are set, and removes the work directory, unless a compiler
# left a temporary file there. After a build that failed or was interrupted it prints one line about
# what is at the program's path, so that no one mistakes an earlier cp_verify for this build's.
# After SIGKILL nothing is printed and the work directory stays.
finish() {
  local status=$?
  set +e # go on to the end even if a removal fails
  in_work rm -f -- cp_verify && [[ $(stat -c %d:%i -- "$work") == "$work_id" ]] && rmdir -- "$work" ||
    echo "build.sh: the work directory $(printf '%q' "$work") could not be removed: it is not empty, or it is no" \
      "longer the directory that this build created. The build does not need it any more." >&2
  [[ $status -eq 0 ]] && return
  if [[ -e $out/cp_verify || -L $out/cp_verify ]]; then
    echo "build.sh: the build failed (status $status). The cp_verify in $where is unchanged and is not from this" \
      "build. See the messages above, if any, and build again." >&2
  else
    echo "build.sh: the build failed (status $status). There is no cp_verify in $where. See the messages above, if" \
      "any, and build again." >&2
  fi
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
cd "$here/src"
export LC_ALL=C
build_word=$({ sha256sum * ../../prover/*/plan_rounds.h ../../prover/hiding/token_plan.h; cd ../../blocks && find . -type f -name '*.h' ! -path '*/tests/*' | sort | xargs sha256sum; } | sha256sum | cut -c1-64)
echo "build=$build_word dw=$dw"
echo "compiling cp_verify.cpp: one compiler call of one to two minutes, which prints nothing while it runs"
# The source digest (`build_word`) is the SHA-256 of sha256sum's output for the files listed above:
# at least every file of this repository that cp_verify.cpp includes.
# A test that is not in this repository checks that, without -DSERVING_BUILD=1, and that
# verifiers_word() in prover/build.sh hashes the same list: change both together.
# The files under prover/ are the sub-proofs' lists of rounds, which prover and verifier share, and
# a header that one of them includes. LC_ALL=C fixes the files' order.
# cp_verify gets the digest as the macro VF_SOURCE_DIGEST and prints it as build=. The digest covers only
# the sources: it tells neither the compiler nor the flags.
# `dw`: 32-bit words in a hash digest (DIGEST_WORDS); verifier/src/vf_digest_words.h compiles only
# with 8.
# -fno-ipa-modref works around a bug in g++ 13.3.0, which can silently delete the stores that fill
# an object passed by reference; verifier/tests/compiler_pass_control.sh demonstrates it.
# It was seen in a CPU prover, reference_prover, built with -O1 -fno-inline, and not at plain -O2.
# That program is not in this repository. clang++ has neither the ipa-modref pass nor the flag.
cxx=(g++ -O2 -fno-ipa-modref)
[[ $clang == 1 ]] && cxx=(clang++ -O2)
# Hardening, after the OpenSSF's "Compiler Options Hardening Guide for C and C++". cp_verify reads
# untrusted files. With these flags an index past the end of a vector or string, a copy past the
# end of an array whose size the compiler knows, or a stack buffer overrun that reaches the return
# address kills the process with a signal, instead of letting it run on with corrupted memory.
# cp_verify runs its stages as child processes, and a stage that a signal kills counts as not passed
# (cpStagesRun() in verifier/src/cp_verify.cpp).
# A local variable that the source leaves uninitialized is zero. Some systems' compilers set some
# of these flags by default; this build does not rely on that.
# What _GLIBCXX_ASSERTIONS costs under g++ 13.3.0 was measured only on another build, and the
# figures are not in this repository. verifier/README.md, "Build", says that the cost in running
# time has not been measured on this program.
# -DSERVING_BUILD=1 is meant to keep test code out of this program. blocks/fence.h
# lists macros that enable test code;
# under this flag it stops the compilation if one is already defined, as one
# given on the command line is. It does not catch a macro that a source file defines later
# (the file states this limit: after the source has included blocks/base.h).
# -DBLK_COUNT_HASH_CALLS=1: the hash in blocks/ counts its calls, for the number of hash calls that
# cp_verify prints.
harden=(-D_GLIBCXX_ASSERTIONS -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -fstack-protector-strong -fstack-clash-protection
  -ftrivial-auto-var-init=zero -fPIE -pie -Wl,-z,relro,-z,now,-z,noexecstack)
# macOS: Apple's linker takes none of the ELF -z options, and every program there is position
# independent already.
if [[ $(uname -s) == Darwin ]]; then
  harden=(-D_GLIBCXX_ASSERTIONS -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -fstack-protector-strong -ftrivial-auto-var-init=zero -fPIE)
fi
[[ $(uname -m) == x86_64 ]] && harden+=(-fcf-protection=full)
"${cxx[@]}" "${harden[@]}" -std=c++17 -Wall -Wextra -Wuninitialized -Werror=unused-result \
  -Wno-misleading-indentation -pthread -I. -I../../blocks -I../../prover -DBLK_FIELD=$field -DBLK_COUNT_HASH_CALLS=1 -DSERVING_BUILD=1 -DDIGEST_WORDS=$dw -DVF_SOURCE_DIGEST="\"$build_word\"" -o "$work/cp_verify" \
  cp_verify.cpp
# Only the rename, at most two lines of output and finish() are left. SIGINT, SIGTERM, SIGHUP and
# SIGPIPE are ignored from here on, so that none of them ends the script with a failure status after
# the rename.
trap '' INT TERM HUP PIPE
had=0 # 1 if a cp_verify, or a symbolic link of that name, is there
[[ -e $out/cp_verify || -L $out/cp_verify ]] && had=1
in_work mv -fT -- cp_verify "$out/cp_verify"
# The build succeeded. "|| true" stops set -e from acting on a failed write.
if [[ $had == 1 ]]; then
  echo "build.sh: overwrote the cp_verify that was already in $where. To keep a cp_verify from elsewhere, build into" \
    "another directory: end the call with \"- DIRECTORY\"" >&2 || true
fi
echo "THE VERIFIER IS BUILT: $(realpath -- "$out/cp_verify")" || true
