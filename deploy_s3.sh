cd ./src

rm chatglm-6b-model.tar.gz
tar zcvf chatglm-6b-model.tar.gz *

aws s3 cp chatglm-6b-model.tar.gz \
  s3://cloudbeer-llm-models/llm/chatglm-6b-model.tar.gz