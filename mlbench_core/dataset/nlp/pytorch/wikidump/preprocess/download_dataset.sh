#! /usr/bin/env bash

set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

COMPRESSED_FILE="wikicorpus_en.xml.bz2"
DECOMPRESSED_FILE="wikicorpus_en.xml"
URL="https://storage.googleapis.com/mlbench-datasets/wikidump/enwiki-20200101-pages-articles-multistream.xml.bz2"
OUTPUT_DIR=${1:-"data"}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DOWNLOAD="${OUTPUT_DIR}/download"
OUTPUT_DIR_EXTRACTED="${OUTPUT_DIR}/extracted"
OUTPUT_DIR_RES="${OUTPUT_DIR}/processed"
mkdir -p $OUTPUT_DIR_DOWNLOAD
mkdir -p $OUTPUT_DIR_EXTRACTED

# Download and decompress
if test -f "${OUTPUT_DIR_DOWNLOAD}/${DECOMPRESSED_FILE}"; then
    echo "Wikidump already decompressed"
else
    if test -f "${OUTPUT_DIR_DOWNLOAD}/${COMPRESSED_FILE}"; then
      echo "Wikidump already downloaded"
  else
    echo "Downloading Wikidump, this may take a while"
    wget -nc -nv -O ${OUTPUT_DIR_DOWNLOAD}/${COMPRESSED_FILE} ${URL}
  fi

  echo "Decompressing Wikidump, this may take a while"
  bzip2 -d ${OUTPUT_DIR_DOWNLOAD}/${COMPRESSED_FILE}
fi

# Clone and run Wiki extractor
pip install wikiextractor
wikiextractor ${OUTPUT_DIR_DOWNLOAD}/${DECOMPRESSED_FILE} -o ${OUTPUT_DIR_EXTRACTED}


# Pre-process data

inputs=${OUTPUT_DIR_EXTRACTED}/"*/wiki_??"
pip install nltk

# Remove doc tag and title
python3 ./cleanup_file.py --data=$inputs --output_suffix='.1'

# Further clean up files
for f in ${inputs}; do
  ./clean.sh ${f}.1 ${f}.2
done

# Sentence segmentation
python3 ./do_sentence_segmentation.py --data=$inputs --input_suffix='.2' --output_suffix='.3'

mkdir -p ${OUTPUT_DIR_RES}

## Choose file size method or number of packages by uncommenting only one of the following do_gather options
# Gather into fixed size packages
python3 ./do_gather.py --data=$inputs --input_suffix='.3' --block_size=26.92 --out_dir=${OUTPUT_DIR_RES}