# Deploy RF-DETR to Google Cloud Run

This deployment runs the native four-class **RF-DETR Nano** checkpoint on CPU.
It scales to zero, accepts one inference request at a time, and permits at most
one instance. These defaults avoid idle compute charges and put a hard ceiling
on parallel compute use.

## Prerequisites

1. Select or create a Google Cloud project with billing enabled.
2. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and
   reopen PowerShell.
3. From the repository root, sign in:

   ```powershell
   gcloud auth login
   ```

4. Ensure the four-class checkpoint exists. The expected SHA-256 is
   `033e2c16e32cc41a8e32cfea73473726c4bb05483110a2cf05f335700eeb1809`.

   ```powershell
   .\.venv\Scripts\python.exe scripts\project_four_class_rfdetr.py
   Get-FileHash models\cloud\rfdetr-nano_v4_four_class.pth -Algorithm SHA256
   ```

## Deploy

Choose a globally unique Google Cloud project ID, then run:

```powershell
.\cloud\deploy-gcp.ps1 -ProjectId YOUR_PROJECT_ID
```

The script enables Cloud Run, Cloud Build, and Artifact Registry; creates a
private Docker repository; builds the CPU image remotely; deploys it in
Singapore; and prints the service URL plus a generated API key. Save the key.
The Cloud Run URL is public so the desktop GUI can reach it, but `/detect`
rejects requests without that application key.

The first request after the service scales to zero can take substantially
longer because it must load PyTorch and RF-DETR. Use `/health` first when
preparing a demonstration.

## Verify

```powershell
$url = 'PASTE_SERVICE_URL'
$key = 'PASTE_API_KEY'
curl.exe "$url/health"
curl.exe -X POST "$url/detect" -H "Content-Type: image/jpeg" -H "X-API-Key: $key" --data-binary "@golden/images/YOUR_IMAGE.jpg"
```

The health response must report `"model":"rfdetr-nano"` and exactly these
classes in order: Escalator, People, Stair, Wet Floor Sign.

Run the desktop GUI against the service using a Python environment with the
[local app dependencies](../README.md#local-setup-and-launch). If you have the
local four-class PyTorch checkpoint, use:

```powershell
.\.venv\Scripts\python.exe src\webapp.py --port 5001 --no-depth --endpoint $url --api-key $key
```

The bundled four-class YOLOv8n ONNX model also works as the local detector.
Install `requirements-onnx.txt` into the GUI environment and launch with:

```powershell
$env:VISIONAID_API_KEY = $key
.\.venv\Scripts\python.exe src\webapp.py --port 5001 --no-depth --endpoint $url --model deploy/vercel/model/yolov8n_v4_four_class.onnx
```

Open `http://127.0.0.1:5001`, then click **Cloud** and confirm the badge reads
**CLOUD, RF-DETR Nano**. Under **Upload → Images**, choose
`demo/Escalator/Escalator landing.jpg` to check for an Escalator detection.
If the service has scaled to zero, call `/health` before switching modes to
give the model time to load.

## Cost controls and cleanup

The deployment uses request-based CPU billing, `min=0`, `max=1`, two vCPUs and
4 GiB RAM. For deadline/demo traffic this should fit comfortably within a $46
credit balance; no GPU or $300 trial is required. Cloud Build and Artifact
Registry also have small usage charges after their free allowances.

Create a billing budget alert in Google Cloud Console at $10, $25, and $40.
A budget is an alert, not an automatic spending cap. To stop all inference
costs immediately, delete the service:

```powershell
gcloud run services delete visionaid-rfdetr --region asia-southeast1
```

To remove the stored image as well:

```powershell
gcloud artifacts repositories delete visionaid --location asia-southeast1
```
