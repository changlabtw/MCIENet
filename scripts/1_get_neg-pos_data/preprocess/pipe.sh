#!/bin/bash
# Script to preprocess ChIA-PET interaction data and generate negative samples
# The input data should have been filtered against blacklisted regions
set -e # Exit script immediately if any command fails

usage()
{
  echo "$(basename "$0") [-h] INTERS DNASE TFPEAKS NAME DATADIR"
  echo "-- Progam to preprocess the interactions and generate negative samples."

  echo "where:"
  echo "-h           show this help text"
  echo "INTERS       Interaction file in BEDPE format"
  echo "DNASE        Dnase/open chromatin regions in BED format"
  echo "TFPEAKS      The transcription factor peaks for the ChIA-PET protein in BED format"
  echo "NAME         The prefix/name for the sample/experiment"
  echo "DATADIR      Location of the output directory"
}

# --------------------- Environment Setup ---------------------
# Add the preprocess directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts/1_get_neg-pos_data" 
echo "Python path: $PYTHONPATH"
# ------------------------------------------------------------

if [ "$1" != "" ]; then
    case $1 in
        -h | --help )           usage
                                exit
                                ;;
    esac
fi

if [ $# -lt 5 ]; then
  usage
  exit
fi

# Assign command line arguments to variables
INTERS=$1      # Input interaction file in BEDPE format
DNASE=$2       # DNase/open chromatin regions in BED format
TFPEAKS=$3     # Transcription factor peaks in BED format
NAME=$4        # Sample/experiment name prefix
DATADIR=$5     # Output directory path

mkdir -p "$DATADIR"

DIR=$(dirname "$0")
echo $DIR

# --------------------- Data Processing ---------------------
# Filter out invalid interactions where anchors overlap or are on different chromosomes
echo "Removing interactions whose two anchors are overlapping or on different chromosomes"
cat $INTERS | awk '$1==$4 && ($3<$5 || $6<$2)' > ${DATADIR}/${NAME}.std.bedpe
bash ${DIR}/process_pos.sh ${DATADIR}/${NAME}.std.bedpe $DNASE 500 $NAME ${DATADIR}

# Generate various types of negative sample pairs
echo "Generating random anchor pairs"
python ${DIR}/generate_random_anchor_pairs.py $NAME $DATADIR

echo "Generating random TF peak pairs"
python ${DIR}/generate_random_pairs_bed.py $NAME $TFPEAKS tf ${DATADIR}

echo "Generating random DNase pairs"
python ${DIR}/generate_random_pairs_bed.py $NAME $DNASE dnase ${DATADIR}

# Filter TF peak pairs to remove any that overlap with existing interactions
echo "Filtering TF peak pairs"
pairToPair -a ${DATADIR}/${NAME}.random_tf_peak_pairs.bedpe -b $INTERS -type notboth \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.no_intra_all.negative_pairs.bedpe -type notboth \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.only_intra_all.negative_pairs.bedpe -type notboth \
    | uniq > ${DATADIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe


# Filter DNase pairs with multiple filtering steps
echo "Filtering DNase pairs"
pairToPair -a ${DATADIR}/${NAME}.shuffled_neg_anchor.neg_pairs.bedpe -b $INTERS -type notboth \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.no_intra_all.negative_pairs.bedpe -type notboth \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.only_intra_all.negative_pairs.bedpe -type notboth \
    | uniq \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe -type notboth \
    | uniq > ${DATADIR}/${NAME}.shuffled_neg_anchor.neg_pairs.filtered.tf_filtered.bedpe

# Generate negative samples and create final dataset
echo "Sampling 5x negative samples"
python ${DIR}/generate_5fold_neg.py $NAME ${DATADIR}

# Combine all negative samples and remove duplicates
echo "Generating extended dataset of negative samples"
cat ${DATADIR}/${NAME}.{only,no}_intra_all.negative_pairs.bedpe \
    ${DATADIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe \
    | pairToPair -a stdin -b ${DATADIR}/${NAME}.neg_pairs_5x.from_singleton_inter_tf_random.bedpe -type notboth \
    | sort -u > ${DATADIR}/${NAME}.extended_negs_with_intra.bedpe
# -------------------- End of Data Processing --------------------
