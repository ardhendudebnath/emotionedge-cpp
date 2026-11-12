#!/usr/bin/env bash
# Runs a command and, if it fails, twice more after a pause: for the CI steps that download.
# - CMake's FetchContent (cmake/) does not retry an HTTP error, so one 5xx from GitHub fails the
#   configure. A failed fetch leaves no stamp, so configuring again fetches again, and the
#   archives are hash-pinned: a retry cannot change what is built.
# - apt and curl, when a mirror or DNS is briefly down.
# A real error fails all three attempts, about a minute later than it would have.
#
#   bash .github/retry.sh cmake -S . -B build
#   RETRY_PAUSE=0 bash .github/retry.sh false   # pause in seconds before the 2nd try, doubled before the 3rd
set -u
attempts=3
pause=${RETRY_PAUSE:-20}
for ((i = 1; ; i++)); do
    "$@" && exit 0
    status=$?
    if ((i == attempts)); then
        echo "::error::failed $attempts times (exit $status): $*"
        exit "$status"
    fi
    echo "::warning::attempt $i of $attempts failed (exit $status), retrying in $((pause * i)) s: $*"
    sleep $((pause * i))
done
