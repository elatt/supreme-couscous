import subprocess
import os

from datarobot_drum import RuntimeParameters

print("\nDownloading model from S3...\n")
src = RuntimeParameters.get("s3Url")
credential = RuntimeParameters.get("s3Credential")

new_env = os.environ.copy()
new_env["AWS_ACCESS_KEY_ID"] = credential["awsAccessKeyId"]
new_env["AWS_SECRET_ACCESS_KEY"] = credential["awsSecretAccessKey"]
if credential.get("awsSessionToken"):
    new_env["AWS_SESSION_TOKEN"] = credential["awsSessionToken"]

subprocess.run(["aws", "s3", "cp", "--recursive", src, "/model-store/"], check=True, env=new_env)
print("\nDONE.\n")
