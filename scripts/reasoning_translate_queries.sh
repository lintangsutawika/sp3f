#!/bin/bash

# Environment Variables
# Put your OpenAI or other LLM API credentials in the .env file
# LLM_API_URL=... this is the base URL for the LLM API
# LLM_API_KEY=... this is the API key for the LLM API
. .env

while getopts ":m:x:y:l:t:o:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    x ) MODEL_PATH=$OPTARG;;
    y ) DATA_PATH=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')

yeval \
    --model ${MODEL_PATH}$MODEL \
    --task ${TASK}_problemt//${LANGUAGE}_translate \
    --include_path tasks/ \
    --api_base ${LLM_API_URL} \
    --api_key ${LLM_API_KEY} \
    --run_name $TASK+$LANGUAGE+translated+queries \
    --sample_args n=1,temperature=1.0 \
    --trust_remote_code \
    --output_path ${DATA_PATH}data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS
