import { ChatOpenAI } from "@langchain/openai";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
// Create the LLM
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const messages = [
  new SystemMessage(
    "You are a friendly professional chef. You are teaching a class on how to make a tasty meal. Answer all question in Japanese."
  ),
  new HumanMessage(
    "子どものためにオムライスを作りたいです。エスニック料理が好きです。"
  ),
];

// recipe for tasteful meal
const schema = z
  .object({
    reasoning: z
      .string()
      .describe(
        "Think carefully about the user's request. Write down your thoughts in a few sentences."
      ),
    description: z.string().describe("description of the meal"),
    ingredients: z.array(
      z.object({
        name: z.string().describe("name of the ingredient"),
        amount: z.string().describe("amount of the ingredient"),
      })
    ),
    instructions: z.array(z.string()).describe("instructions to make the meal"),
    name: z.string().describe("name of the meal"),
  })
  .describe("recipe for a tasteful meal");

async function main() {
  const result = await llm.withStructuredOutput(schema).invoke(messages);
  console.log(JSON.stringify(result, null, 2));
}

main();
