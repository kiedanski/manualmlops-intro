curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -d '{
        "instances": [
          {"content": "$@"}
        ]
      }' \
https://northamerica-northeast1-aiplatform.googleapis.com/v1/projects/795588112930/locations/northamerica-northeast1/endpoints/3003892155363098624:predict
