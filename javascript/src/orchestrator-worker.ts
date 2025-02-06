import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import path from "path";
import { z } from "zod";
import { traceable } from "langsmith/traceable";
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const model = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
});

const parser = new StringOutputParser();

const ORCHESTRATOR_PROMPT = `
Analyze this task and break it down into 2-3 distinct approaches:

Task: {task}

Provide an Analysis:

Explain your understanding of the task and which variations would be valuable.
Focus on how each approach serves different aspects of the task.

Along with the analysis, provide 2-3 approaches to tackle the task, each with a brief description:

Formal style: Write technically and precisely, focusing on detailed specifications
Conversational style: Write in a friendly and engaging way that connects with the reader
Hybrid style: Tell a story that includes technical details, combining emotional elements with specifications

Return only JSON output.
`;

const WORKER_PROMPT = `
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return only your response:
[Your content here, maintaining the specified style and fully addressing requirements.]
`;

async function orchestrator_worker(task: string) {
  async function run(task: string) {
    const prompt = PromptTemplate.fromTemplate(ORCHESTRATOR_PROMPT);
    const schema = z.object({
      analysis: z.string(),
      tasks: z.array(
        z.object({
          reasoning: z.string().describe("The reasoning for the task"),
          type: z
            .enum(["Formal", "Conversational", "Hybrid"])
            .describe("The type of the task"),
          description: z
            .string()
            .describe(
              "The description of the task and how it serves the original task"
            ),
        })
      ),
    });

    const chain = prompt.pipe(model.withStructuredOutput(schema));
    const result = await chain.invoke({ task });
    type Task = z.infer<typeof schema>["tasks"][number];
    async function worker(t: Task) {
      const workerPrompt = PromptTemplate.fromTemplate(WORKER_PROMPT);
      const workerChain = workerPrompt.pipe(model).pipe(parser);
      return workerChain.invoke({
        original_task: task,
        task_type: t.type,
        task_description: t.description,
      });
    }
    const workerResults = await Promise.all(result.tasks.map(worker));
    return workerResults;
  }
  return traceable(run, { name: "orchestrator_worker" })(task);
}

async function main() {
  const task =
    "Write a product description for a new eco-friendly water bottle. The target_audience is environmentally conscious millennials and key product features are: plastic-free, insulated, lifetime warranty";
  const result = await orchestrator_worker(task);
  for (const workerResult of result) {
    console.log(workerResult);
  }
}

main();
