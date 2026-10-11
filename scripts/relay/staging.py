#!/usr/bin/env python3
"""Create/delete an isolated, bounded protocol-3 WebSocket UAT stack only."""
import argparse
import json
from pathlib import Path
import time
import deploy

NAME = "mold-relay-v3-uat"


def template():
    ref = lambda key: {"Ref": key}
    get = lambda key, attr: {"Fn::GetAtt": [key, attr]}
    sub = lambda value: {"Fn::Sub": value}
    runtime_keys = ["v3-connection#*", "v3-host#*", "v3-policy#*", "v3-invite#*", "v3-pending", "v3-pending#*", "v3-hosts"]
    resources = {
        "Sessions": {"Type": "AWS::DynamoDB::Table", "Properties": {"TableName": NAME + "-sessions", "BillingMode": "PAY_PER_REQUEST", "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}], "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}], "TimeToLiveSpecification": {"AttributeName": "expiresAt", "Enabled": True}, "Tags": [{"Key": "BillingGroup", "Value": "urandom.io"}, {"Key": "Purpose", "Value": "MoldV3TemporaryUAT"}]}},
        "Api": {"Type": "AWS::ApiGatewayV2::Api", "Properties": {"Name": NAME, "ProtocolType": "WEBSOCKET", "RouteSelectionExpression": "$request.body.a"}},
        "Logs": {"Type": "AWS::Logs::LogGroup", "Properties": {"LogGroupName": "/aws/lambda/" + NAME + "-router", "RetentionInDays": 1}},
        "Role": {"Type": "AWS::IAM::Role", "Properties": {"RoleName": NAME + "-router", "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}, "Policies": [{"PolicyName": NAME, "PolicyDocument": {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": get("Logs", "Arn")},
            {"Effect": "Allow", "Action": ["dynamodb:GetItem"], "Resource": get("Sessions", "Arn"), "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": runtime_keys + ["v3-registration#*"]}}},
            {"Effect": "Allow", "Action": ["dynamodb:PutItem", "dynamodb:DeleteItem"], "Resource": get("Sessions", "Arn"), "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": runtime_keys}}},
            {"Effect": "Allow", "Action": "execute-api:ManageConnections", "Resource": sub("arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${Api}/uat/*/@connections/*")}
        ]}}]}},
        "Router": {"Type": "AWS::Lambda::Function", "Properties": {"FunctionName": NAME + "-router", "Runtime": "nodejs22.x", "Handler": "index.handler", "Architectures": ["arm64"], "MemorySize": 256, "Timeout": 10, "ReservedConcurrentExecutions": 32, "Role": get("Role", "Arn"), "Code": {"ZipFile": "exports.handler=async()=>({statusCode:503});"}, "Environment": {"Variables": {"TABLE_NAME": ref("Sessions"), "MANAGEMENT_ENDPOINT": sub("https://${Api}.execute-api.${AWS::Region}.amazonaws.com/uat")}}}},
        "Integration": {"Type": "AWS::ApiGatewayV2::Integration", "Properties": {"ApiId": ref("Api"), "IntegrationType": "AWS_PROXY", "IntegrationMethod": "POST", "IntegrationUri": sub("arn:${AWS::Partition}:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${Router.Arn}/invocations")}},
        "Permission": {"Type": "AWS::Lambda::Permission", "Properties": {"FunctionName": ref("Router"), "Action": "lambda:InvokeFunction", "Principal": "apigateway.amazonaws.com", "SourceArn": sub("arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${Api}/*/*")}},
        "Stage": {"Type": "AWS::ApiGatewayV2::Stage", "DependsOn": ["Connect", "Disconnect", "Default"], "Properties": {"ApiId": ref("Api"), "StageName": "uat", "AutoDeploy": True, "DefaultRouteSettings": {"ThrottlingBurstLimit": 256, "ThrottlingRateLimit": 800}}},
    }
    for label, route in [("Connect", "$connect"), ("Disconnect", "$disconnect"), ("Default", "$default")]:
        resources[label] = {"Type": "AWS::ApiGatewayV2::Route", "Properties": {"ApiId": ref("Api"), "RouteKey": route, "Target": {"Fn::Join": ["/", ["integrations", ref("Integration")]]}}}
    return {"AWSTemplateFormatVersion": "2010-09-09", "Description": "Temporary Mold v3 encrypted relay UAT; synthetic data only; delete after proof.", "Resources": resources, "Outputs": {"WebSocketUrl": {"Value": sub("wss://${Api}.execute-api.${AWS::Region}.amazonaws.com/uat")}, "Function": {"Value": ref("Router")}, "Table": {"Value": ref("Sessions")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "update", "delete", "status", "template"))
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
        result = deploy.aws(["cloudformation", "create-stack", "--stack-name", NAME, "--template-body", f"file://{path}", "--capabilities", "CAPABILITY_NAMED_IAM", "--tags", "Key=BillingGroup,Value=urandom.io", "Key=Purpose,Value=MoldV3TemporaryUAT"], options)
        deploy.private_write(manifest, json.dumps({"stackId": result["StackId"], "profile": options.profile, "region": options.region, "createdAt": int(time.time())}))
        deploy.aws(["cloudformation", "wait", "stack-create-complete", "--stack-name", result["StackId"]], options)
        options.function_prefix = NAME
        deploy.deploy_function(options, "router", options.package)
    saved = json.loads(manifest.read_text())
    if saved["profile"] != options.profile or saved["region"] != options.region or not saved["stackId"].split(":stack/")[-1].startswith(NAME + "/"):
        raise RuntimeError("Staging manifest identity mismatch; refusing action")
    if options.action == "update":
        if not options.package or not deploy.protocol_contract(options.package, "router"):
            raise RuntimeError("Update requires a verified protocol-3 router ZIP")
        options.function_prefix = NAME
        deploy.deploy_function(options, "router", options.package)
    if options.action in ("create", "update"):
        saved["codeSha256"] = deploy.package_hash(options.package)
    if options.action == "delete":
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
