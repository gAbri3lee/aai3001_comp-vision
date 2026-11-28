[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProjectId,

    [string]$Region = 'asia-southeast1',
    [string]$Service = 'visionaid-rfdetr',
    [string]$Repository = 'visionaid',
    [string]$ApiKey
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command gcloud.cmd -ErrorAction SilentlyContinue)) {
    throw 'gcloud is not installed. Install Google Cloud CLI, reopen PowerShell, and run this script again.'
}
$weightsPath = 'models/cloud/rfdetr-nano_v4_four_class.pth'
if (-not (Test-Path -LiteralPath $weightsPath -PathType Leaf)) {
    throw 'Missing four-class RF-DETR checkpoint. Run: .\.venv\Scripts\python.exe scripts\project_four_class_rfdetr.py'
}
$uploadFiles = gcloud.cmd meta list-files-for-upload
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Cloud Build upload.' }
$uploadFiles = $uploadFiles -replace '\\', '/'
if ($uploadFiles -notcontains $weightsPath) {
    throw "Cloud Build would skip $weightsPath. Check .gcloudignore before retrying."
}
if (-not $ApiKey) {
    $bytes = New-Object byte[] 32
    $random = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $random.GetBytes($bytes)
    } finally {
        $random.Dispose()
    }
    $ApiKey = [BitConverter]::ToString($bytes).Replace('-', '').ToLowerInvariant()
}

if ($ProjectId -match '^\d+$') {
    $resolvedProjectId = gcloud.cmd projects describe $ProjectId --format 'value(projectId)'
    if ($LASTEXITCODE -ne 0 -or -not $resolvedProjectId) {
        throw "Could not resolve project number $ProjectId. Check the project ID in Google Cloud Console."
    }
    $ProjectId = $resolvedProjectId.Trim()
}

gcloud.cmd config set project $ProjectId
if ($LASTEXITCODE -ne 0) { throw 'Could not select the Google Cloud project.' }
gcloud.cmd services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $ProjectId
if ($LASTEXITCODE -ne 0) { throw 'Could not enable the required Google Cloud APIs.' }

$repositories = gcloud.cmd artifacts repositories list --project $ProjectId --location $Region --format 'value(name.basename())'
if ($LASTEXITCODE -ne 0) { throw 'Could not list Artifact Registry repositories.' }
if ($repositories -notcontains $Repository) {
    gcloud.cmd artifacts repositories create $Repository --project $ProjectId --repository-format docker --location $Region --description 'VisionAid RF-DETR images'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Artifact Registry repository.' }
}

$substitutions = "_REGION=$Region,_REPOSITORY=$Repository,_IMAGE=$Service"
gcloud.cmd builds submit --project $ProjectId --region $Region --config cloud/cloudbuild.yaml --substitutions $substitutions .
if ($LASTEXITCODE -ne 0) { throw 'Cloud Build failed; the service was not deployed.' }

$image = "$Region-docker.pkg.dev/$ProjectId/$Repository/${Service}:latest"
gcloud.cmd run deploy $Service `
    --project $ProjectId `
    --image $image `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 8080 `
    --cpu 2 `
    --memory 4Gi `
    --concurrency 1 `
    --min 0 `
    --max 1 `
    --timeout 120 `
    --set-env-vars "VISIONAID_DEVICE=cpu,VISIONAID_API_KEY=$ApiKey"
if ($LASTEXITCODE -ne 0) { throw 'Cloud Run deployment failed.' }

$url = gcloud.cmd run services describe $Service --project $ProjectId --region $Region --format 'value(status.url)'
if ($LASTEXITCODE -ne 0 -or -not $url) { throw 'The service deployed, but its URL could not be read.' }
Write-Host "Service URL: $url"
Write-Host "API key (save this now): $ApiKey"
Write-Host "Health check: curl.exe $url/health"
Write-Host "Start GUI: .\.venv\Scripts\python.exe src\webapp.py --port 5001 --no-depth --endpoint $url --api-key $ApiKey"
