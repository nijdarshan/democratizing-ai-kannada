$PROJECT_ID = "utopian-saga-438818-j0"
$MODEL = "claude-3-5-sonnet-v2@20241022"
$LOCATION = "us-east5"

# Get the access token
$ACCESS_TOKEN = & gcloud auth print-access-token

# Make the API call
$headers = @{
    "Authorization" = "Bearer $ACCESS_TOKEN"
    "Content-Type" = "application/json; charset=utf-8"
}

Invoke-RestMethod `
    -Uri "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/publishers/anthropic/models/$MODEL:rawPredict" `
    -Method Post `
    -Headers $headers `
    -InFile "request.json"