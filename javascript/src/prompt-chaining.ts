import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from "@langchain/core/messages";
import { traceable } from "langsmith/traceable";

import * as dotenv from "dotenv";
import path from "path";
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const model = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
});

const parser = new StringOutputParser();

type PromptChainingInput = {
    inputQuery: string;
    prompts: string[];
}
async function prompt_chaining({ inputQuery, prompts }: PromptChainingInput) {
    async function run() {
        const messages: BaseMessage[] = [
            new SystemMessage({
                content: "You are a helpful assistant that can solve math problems.",
            }),
            new HumanMessage(inputQuery),
        ];
    
        let response = inputQuery;
        for (const prompt of prompts) {
            messages.push(new HumanMessage(prompt));
            const chain = model.pipe(parser);
            response = await chain.invoke(messages);
            messages.push(new AIMessage(response));
        }
        return response;
    }
    return traceable(run, {
        name: "prompt_chaining",
    })();
}


async function main() {
    const question = "Sally earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?";

    const promptChain = [
      "Given the math problem, ONLY extract any relevant numerical information and how it can be used.",
      "Given the numerical information extracted, ONLY express the steps you would take to solve the problem.",
      "Given the steps, express the final answer to the problem."
    ];
    const response = await prompt_chaining({ inputQuery: question, prompts: promptChain });
    console.log(response);
}

// main();

export { prompt_chaining };