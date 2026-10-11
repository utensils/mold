import {
  DynamoDBClient,
  GetItemCommand,
  PutItemCommand,
  DeleteItemCommand,
} from "@aws-sdk/client-dynamodb";
import { encodeDocument, decodeDocument } from "./store-codec.mjs";
const dynamo = new DynamoDBClient({});
export const store = {
  async get(id) {
    const r = await dynamo.send(
      new GetItemCommand({
        TableName: process.env.TABLE_NAME,
        Key: { id: { S: id } },
        ConsistentRead: true,
      }),
    );
    return r.Item
      ? {
          ...decodeDocument(id, r.Item),
          revision: Number(r.Item.revision?.N ?? 0),
        }
      : undefined;
  },
  async cas(id, revision, value) {
    try {
      await dynamo.send(
        new PutItemCommand({
          TableName: process.env.TABLE_NAME,
          Item: {
            id: { S: id },
            ...encodeDocument(id, value),
            revision: { N: String(revision + 1) },
            expiresAt: {
              N: String(
                value.expiresAt ?? Math.floor(Date.now() / 1000) + 3600,
              ),
            },
          },
          ConditionExpression:
            revision === 0
              ? "attribute_not_exists(id)"
              : "revision = :revision",
          ...(revision
            ? {
                ExpressionAttributeValues: {
                  ":revision": { N: String(revision) },
                },
              }
            : {}),
        }),
      );
      return true;
    } catch (e) {
      if (e.name === "ConditionalCheckFailedException") return false;
      throw e;
    }
  },
  async put(id, value) {
    await dynamo.send(
      new PutItemCommand({
        TableName: process.env.TABLE_NAME,
        Item: {
          id: { S: id },
          ...encodeDocument(id, value),
          revision: { N: "1" },
          expiresAt: {
            N: String(value.expiresAt ?? Math.floor(Date.now() / 1000) + 3600),
          },
        },
      }),
    );
  },
  async remove(id) {
    await dynamo.send(
      new DeleteItemCommand({
        TableName: process.env.TABLE_NAME,
        Key: { id: { S: id } },
      }),
    );
  },
};
