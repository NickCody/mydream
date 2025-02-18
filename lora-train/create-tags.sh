#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Usage message
usage() {
    echo "Usage: $0 -t \"tags\" <filespec>"
    exit 1
}

# Parse arguments
while getopts ":t:" opt; do
    case ${opt} in
        t )
            TAGS=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Ensure a filespec is provided
if [ -z "$1" ]; then
    echo "Error: Missing filespec argument."
    usage
fi

FILESPEC=$1

# Get absolute path of the current working directory
DIR=$(pwd)

# Process each file in the specified pattern
for file in "$DIR"/$FILESPEC; do
    if [ -f "$file" ]; then
        base_name="${file%.*}"
        txt_file="${base_name}.txt"
        echo "$TAGS" > "$txt_file"
        echo "Created: $txt_file with tags: $TAGS"
    fi
done
