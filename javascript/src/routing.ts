import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { traceable } from "langsmith/traceable";
import * as dotenv from "dotenv";
import path from "path";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const model = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
});

const parser = new StringOutputParser();

type AssistantInput = {
  id: string;
  description: string;
  prompt: string;
};

class Assistant {
  id: string;
  description: string;
  prompt: string;

  constructor(id: string, description: string, prompt: string) {
    this.id = id;
    this.description = description;
    this.prompt = prompt;
  }

  async run(input: string) {
    const messages: BaseMessage[] = [
      new SystemMessage({
        content: this.prompt,
      }),
      new HumanMessage(input),
    ];
    const chain = model.pipe(parser);
    const response = await chain.invoke(messages);
    return response;
  }

  static from(input: AssistantInput) {
    return new Assistant(input.id, input.description, input.prompt);
  }
}

type WorkflowInput = {
  assistants: Assistant[];
};

class Workflow {
  assistants: Assistant[];

  constructor(assistants: Assistant[]) {
    this.assistants = assistants;
  }

  private async _run(task: string) {
    const routerSchema = z.object({
      reason: z.string().describe("The reason for the choice"),
      assistant_id: z.string().describe(`The ID of the assistant to use. Choose from: ${this.assistants.map((assistant) => assistant.id).join(", ")}`),
    });
    const routerPrompt = PromptTemplate.fromTemplate(
      "Given a user prompt/query: {user_query}, select the best option out of the following routes:\n{routes}\nAnswer only in JSON format."
    );
    const routes = this.assistants
      .map((assistant) => `ID: ${assistant.id}, Description: ${assistant.description}`)
      .join("\n");
    const routerChain = routerPrompt.pipe(
      model.withStructuredOutput(routerSchema, {
        strict: true,
      })
    );
    const response = await routerChain.invoke({
      user_query: task,
      routes,
    });
    const assistant = this.assistants.find(
      (assistant) => assistant.id === response.assistant_id
    );
    if (!assistant) {
      throw new Error(`Assistant with ID ${response.assistant_id} not found`);
    }
    const result = await assistant.run(task);
    return result;
  }

  async run(task: string) {
    return traceable(this._run.bind(this), { name: "routing" })(task);
  }

  static from(input: WorkflowInput) {
    return new Workflow(input.assistants);
  }
}

async function main() {
  const code_generation = Assistant.from({
    id: "code_generation",
    description: "Suited for code generation",
    prompt: "You are a helpful assistant that generates code for a website",
  });
  const trip_planner = Assistant.from({
    id: "trip_planner",
    description: "Suited for trip planning",
    prompt: "You are a helpful assistant that plans trips",
  });
  const story_teller = Assistant.from({
    id: "story_teller",
    description: "Suited for story telling",
    prompt: "You are a helpful assistant that tells stories",
  });
  const workflow = Workflow.from({ assistants: [code_generation, trip_planner, story_teller] });
  const tasks = [
    "Write a Python function to check if a number is prime.",
    "Plan a 2-week trip to Europe.",
    "Tell me a story about a cat.",
  ];
  for (const task of tasks) {
    const result = await workflow.run(task);
    console.log(result);
  }
}

main();
