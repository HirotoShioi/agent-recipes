import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { StructuredTool, tool } from "@langchain/core/tools";
import z from "zod";

// Create the LLM
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const addTool = tool(
  async ({ a, b }) => {
    return a + b;
  },
  {
    name: "add",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
    description: "Adds a and b.",
  }
);

const multiplyTool = tool(
  async ({ a, b }) => {
    return a * b;
  },
  {
    name: "multiply",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
    description: "Multiplies a and b.",
  }
);

const tools = [addTool, multiplyTool];

// Bind tools to the LLM
const llmWithTools = llm.bindTools(tools);

async function main() {
  const query = "What is 3 * 12? Also, what is 11 + 49?";

  const messages = [new HumanMessage(query)];

  // Invoke LLM with initial query
  const aiMsg = await llmWithTools.invoke(messages);

  if (!(aiMsg instanceof AIMessage)) {
    throw new Error("Expected AIMessage but got different type");
  }

  messages.push(aiMsg);

  // Create a map of tools by name
  const toolsByName: Record<string, StructuredTool> = {
    add: addTool,
    multiply: multiplyTool,
  };

  // Process tool calls
  if (aiMsg.tool_calls) {
    for (const toolCall of aiMsg.tool_calls) {
      // @ts-ignore
      const selectedTool = toolsByName[toolCall.name];
      if (!selectedTool) {
        continue;
      }
      const toolMsg: ToolMessage = await selectedTool.invoke(toolCall);
      messages.push(toolMsg);
    }
  }

  // Final LLM invocation
  const result = await llmWithTools.invoke(messages);

  console.log(result.content);
}

// Run the main function
main().catch(console.error);
