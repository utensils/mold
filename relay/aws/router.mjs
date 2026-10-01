import {
  ApiGatewayManagementApiClient,
  PostToConnectionCommand,
  DeleteConnectionCommand,
  GetConnectionCommand,
} from "@aws-sdk/client-apigatewaymanagementapi";
import { store, parameter } from "./aws-store.mjs";
import { createRouter } from "./router-core.mjs";
const api = new ApiGatewayManagementApiClient({
  endpoint: process.env.MANAGEMENT_ENDPOINT,
});
export const handler = createRouter({
  store,
  tokens: async () => ({
    host: await parameter(process.env.HOST_TOKEN_PARAMETER),
    frontend: await parameter(process.env.BRIDGE_TOKEN_PARAMETER),
  }),
  post: async (id, frame) => {
    await api.send(
      new PostToConnectionCommand({
        ConnectionId: id,
        Data: Buffer.from(JSON.stringify(frame)),
      }),
    );
  },
  checkConnection: async (id) => {
    await api.send(new GetConnectionCommand({ ConnectionId: id }));
  },
  close: async (id) => {
    try {
      await api.send(new DeleteConnectionCommand({ ConnectionId: id }));
    } catch (e) {
      if (e.$metadata?.httpStatusCode !== 410) throw e;
    }
  },
});
