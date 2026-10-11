#!/usr/bin/env python3
"""Create/delete an isolated, bounded protocol-3 WebSocket UAT stack only."""
import argparse
import json
from pathlib import Path
import time
import deploy

NAME = "mold-relay-v3-uat"
TAGS={'BillingGroup':'urandom.io','Project':'mold','Service':'mold-relay','Environment':'UAT','ManagedBy':'CloudFormation','Purpose':'MoldV3TemporaryUAT'}


def template():
    ref = lambda key: {"Ref": key}
    get = lambda key, attr: {"Fn::GetAtt": [key, attr]}
    sub = lambda value: {"Fn::Sub": value}
    runtime_keys = ["v3-connection#*", "v3-host#*", "v3-policy#*", "v3-invite#*", "v3-pending", "v3-pending#*", "v3-hosts", "v3-objects#*"]
    resources = {
        "Sessions": {"Type": "AWS::DynamoDB::Table", "Properties": {"TableName": NAME + "-sessions", "BillingMode": "PAY_PER_REQUEST", "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}], "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}], "TimeToLiveSpecification": {"AttributeName": "expiresAt", "Enabled": True}, "Tags": [{"Key": "BillingGroup", "Value": "urandom.io"}, {"Key": "Purpose", "Value": "MoldV3TemporaryUAT"}]}},
        "Api": {"Type": "AWS::ApiGatewayV2::Api", "Properties": {"Name": NAME, "ProtocolType": "WEBSOCKET", "RouteSelectionExpression": "$request.body.a"}},
        "Ciphertext": {"Type":"AWS::S3::Bucket","Properties":{"BucketName":sub(NAME+'-${AWS::AccountId}-${AWS::Region}'),"OwnershipControls":{"Rules":[{"ObjectOwnership":"BucketOwnerEnforced"}]},"PublicAccessBlockConfiguration":{"BlockPublicAcls":True,"BlockPublicPolicy":True,"IgnorePublicAcls":True,"RestrictPublicBuckets":True},"BucketEncryption":{"ServerSideEncryptionConfiguration":[{"ServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]},"LifecycleConfiguration":{"Rules":[{"Id":"expire-ciphertext","Status":"Enabled","Prefix":"ciphertext/v3/","ExpirationInDays":1,"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":1}}]},"CorsConfiguration":{"CorsRules":[{"AllowedOrigins":["https://utensils.io","http://127.0.0.1:4173","http://127.0.0.1:5174","http://localhost:4173","tauri://localhost","https://tauri.localhost","http://tauri.localhost"],"AllowedMethods":["GET","HEAD","PUT"],"AllowedHeaders":["content-type","x-amz-checksum-sha256","if-none-match","range"],"ExposedHeaders":["etag","x-amz-checksum-sha256","content-length"],"MaxAge":300}]},"Tags":[{"Key":"BillingGroup","Value":"urandom.io"},{"Key":"Purpose","Value":"MoldV3TemporaryUAT"}]}},
        "CiphertextPolicy":{"Type":"AWS::S3::BucketPolicy","Properties":{"Bucket":ref('Ciphertext'),"PolicyDocument":{"Version":"2012-10-17","Statement":[{"Effect":"Deny","Principal":"*","Action":"s3:*","Resource":[get('Ciphertext','Arn'),sub('${Ciphertext.Arn}/*')],"Condition":{"Bool":{"aws:SecureTransport":"false"}}}]}}},
        "Logs": {"Type": "AWS::Logs::LogGroup", "Properties": {"LogGroupName": "/aws/lambda/" + NAME + "-router", "RetentionInDays": 1}},
        "Role": {"Type": "AWS::IAM::Role", "Properties": {"RoleName": NAME + "-router", "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}, "Policies": [{"PolicyName": NAME, "PolicyDocument": {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": get("Logs", "Arn")},
            {"Effect": "Allow", "Action": ["dynamodb:GetItem"], "Resource": get("Sessions", "Arn"), "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": runtime_keys + ["v3-registration#*"]}}},
            {"Effect": "Allow", "Action": ["dynamodb:PutItem", "dynamodb:DeleteItem"], "Resource": get("Sessions", "Arn"), "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": runtime_keys}}},
            {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":sub('${Ciphertext.Arn}/ciphertext/v3/*')},
            {"Effect": "Allow", "Action": "execute-api:ManageConnections", "Resource": sub("arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${Api}/uat/*/@connections/*")}
        ]}}]}},
        "Router": {"Type": "AWS::Lambda::Function", "Properties": {"FunctionName": NAME + "-router", "Runtime": "nodejs22.x", "Handler": "index.handler", "Architectures": ["arm64"], "MemorySize": 256, "Timeout": 10, "ReservedConcurrentExecutions": 32, "Role": get("Role", "Arn"), "Code": {"ZipFile": "exports.handler=async()=>({statusCode:503});"}, "Environment": {"Variables": {"TABLE_NAME": ref("Sessions"), "CIPHERTEXT_BUCKET":ref('Ciphertext'),"MANAGEMENT_ENDPOINT": sub("https://${Api}.execute-api.${AWS::Region}.amazonaws.com/uat")}}}},
        "Integration": {"Type": "AWS::ApiGatewayV2::Integration", "Properties": {"ApiId": ref("Api"), "IntegrationType": "AWS_PROXY", "IntegrationMethod": "POST", "IntegrationUri": sub("arn:${AWS::Partition}:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${Router.Arn}/invocations")}},
        "Permission": {"Type": "AWS::Lambda::Permission", "Properties": {"FunctionName": ref("Router"), "Action": "lambda:InvokeFunction", "Principal": "apigateway.amazonaws.com", "SourceArn": sub("arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${Api}/*/*")}},
        "Stage": {"Type": "AWS::ApiGatewayV2::Stage", "DependsOn": ["Connect", "Disconnect", "Default"], "Properties": {"ApiId": ref("Api"), "StageName": "uat", "AutoDeploy": True, "DefaultRouteSettings": {"ThrottlingBurstLimit": 256, "ThrottlingRateLimit": 800}}},
    }
    for label, route in [("Connect", "$connect"), ("Disconnect", "$disconnect"), ("Default", "$default")]:
        resources[label] = {"Type": "AWS::ApiGatewayV2::Route", "Properties": {"ApiId": ref("Api"), "RouteKey": route, "Target": {"Fn::Join": ["/", ["integrations", ref("Integration")]]}}}
    tags=TAGS
    for item in resources.values():
        if item['Type'] in ('AWS::ApiGatewayV2::Api','AWS::ApiGatewayV2::Stage'):
            item['Properties']['Tags']=tags
        elif item['Type'] in ('AWS::Lambda::Function','AWS::DynamoDB::Table','AWS::Logs::LogGroup','AWS::IAM::Role','AWS::S3::Bucket'):
            item['Properties']['Tags']=[{'Key':k,'Value':v} for k,v in tags.items()]
    return {"AWSTemplateFormatVersion": "2010-09-09", "Description": "Temporary Mold v3 encrypted relay UAT; synthetic data only; delete after proof.", "Resources": resources, "Outputs": {"WebSocketUrl": {"Value": sub("wss://${Api}.execute-api.${AWS::Region}.amazonaws.com/uat")}, "Function": {"Value": ref("Router")}, "Table": {"Value": ref("Sessions")},"CiphertextBucket":{"Value":ref('Ciphertext')}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "configure", "update", "delete", "status", "template"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--profile", default="dev.urandom.io")
    parser.add_argument("--region", default="us-west-2")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = options.output / "stack.json"
    if options.action == "template":
        deploy.private_write(path, json.dumps(template(), indent=2))
        print(path)
        return
    manifest = options.output / "resources.json"
    if options.action == "create":
        if not options.package or not deploy.protocol_contract(options.package, "router"):
            raise RuntimeError("Create requires a verified protocol-3 router ZIP")
        deploy.private_write(path, json.dumps(template(), indent=2))
        result = deploy.aws(["cloudformation", "create-stack", "--stack-name", NAME, "--template-body", f"file://{path}", "--capabilities", "CAPABILITY_NAMED_IAM", "--tags", *[f'Key={k},Value={v}' for k,v in TAGS.items()]], options)
        deploy.private_write(manifest, json.dumps({"stackId": result["StackId"], "profile": options.profile, "region": options.region, "createdAt": int(time.time())}))
        deploy.aws(["cloudformation", "wait", "stack-create-complete", "--stack-name", result["StackId"]], options)
        options.function_prefix = NAME
        deploy.deploy_function(options, "router", options.package)
    saved = json.loads(manifest.read_text())
    if saved["profile"] != options.profile or saved["region"] != options.region or not saved["stackId"].split(":stack/")[-1].startswith(NAME + "/"):
        raise RuntimeError("Staging manifest identity mismatch; refusing action")
    if options.action == 'configure':
        before=deploy.aws(['lambda','get-function-configuration','--function-name',NAME+'-router'],options)['CodeSha256']
        path.write_text(json.dumps(template(),indent=2))
        deploy.aws(['cloudformation','update-stack','--stack-name',saved['stackId'],'--template-body',f'file://{path}','--capabilities','CAPABILITY_NAMED_IAM','--tags',*[f'Key={k},Value={v}' for k,v in TAGS.items()]],options)
        deploy.aws(['cloudformation','wait','stack-update-complete','--stack-name',saved['stackId']],options)
        after=deploy.aws(['lambda','get-function-configuration','--function-name',NAME+'-router'],options)['CodeSha256']
        if before!=after: raise RuntimeError('Staging configuration unexpectedly changed code; stop UAT')
    if options.action == "update":
        if not options.package or not deploy.protocol_contract(options.package, "router"):
            raise RuntimeError("Update requires a verified protocol-3 router ZIP")
        options.function_prefix = NAME
        deploy.deploy_function(options, "router", options.package)
    if options.action in ("create", "update"):
        saved["codeSha256"] = deploy.package_hash(options.package)
    if options.action == "delete":
        bucket=saved.get('outputs',{}).get('CiphertextBucket')
        if bucket:
            expected=f'{NAME}-{saved["stackId"].split(":")[4]}-{options.region}'
            if bucket!=expected: raise RuntimeError('Unexpected staging bucket; refusing cleanup')
            while True:
                page=deploy.aws(['s3api','list-objects-v2','--bucket',bucket,'--prefix','ciphertext/v3/','--max-keys','500','--no-paginate'],options)
                keys=[item['Key'] for item in page.get('Contents',[])]
                if not keys: break
                if any(not key.startswith('ciphertext/v3/') for key in keys): raise RuntimeError('Unexpected staging object prefix')
                result=deploy.aws(['s3api','delete-objects','--bucket',bucket,'--delete',json.dumps({'Objects':[{'Key':key} for key in keys],'Quiet':True})],options)
                if result.get('Errors'):raise RuntimeError('Could not clean synthetic staging ciphertext; details omitted')
        deploy.aws(["cloudformation", "delete-stack", "--stack-name", saved["stackId"]], options)
        deploy.aws(["cloudformation", "wait", "stack-delete-complete", "--stack-name", saved["stackId"]], options)
        print("Deleted isolated v3 UAT stack")
        return
    result = deploy.aws(["cloudformation", "describe-stacks", "--stack-name", saved["stackId"]], options)["Stacks"][0]
    resources = deploy.aws(["cloudformation", "list-stack-resources", "--stack-name", saved["stackId"]], options)["StackResourceSummaries"]
    saved["outputs"] = {item["OutputKey"]: item["OutputValue"] for item in result.get("Outputs", [])}
    saved["resources"] = [{"logicalId": r["LogicalResourceId"], "physicalId": r.get("PhysicalResourceId"), "type": r["ResourceType"]} for r in resources]
    manifest.write_text(json.dumps(saved, indent=2))
    print(json.dumps({"status": result["StackStatus"], **saved["outputs"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
