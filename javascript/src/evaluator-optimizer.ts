import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import path from "path";
import { z } from "zod";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const model = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0,
});

// Schema definitions for structured output
const generatorSchema = z.object({
    thoughts: z.string().describe("Your understanding of the task and feedback and how you plan to improve"),
    code: z.string().describe("Your code implementation here"),
});

type GeneratorResponse = z.infer<typeof generatorSchema>;
const evaluatorSchema = z.object({
    feedback: z.string().describe("Detailed feedback on the code implementation"),
    pass: z.enum(["PASS", "FAIL", "NEEDS_IMPROVEMENT"]).describe("Whether the code implementation passes all criteria"),
});

type EvaluatorResponse = z.infer<typeof evaluatorSchema>;
// Generator function to create code based on task and feedback
async function generateCode(task: string, feedback?: string, code?: string): Promise<GeneratorResponse> {
  const GENERATOR_PROMPT = `
    Your goal is to complete the task based on <user input>. If there are feedback 
    from your previous generations, you should reflect on them to improve your solution
    
    Task:
    ${task}
    `;
  const messages = [new SystemMessage(GENERATOR_PROMPT)];
  
  if (code) {
    messages.push(new HumanMessage(`Code: ${code} \n\n Feedback: ${feedback}`));
  }
  
  return await model.withStructuredOutput(generatorSchema).invoke(messages);
}

// Evaluator function to assess code quality
async function evaluateCode(task: string, code: string): Promise<EvaluatorResponse> {
  const EVALUATOR_PROMPT = `
    Evaluate this following code implementation for:
    1. code correctness
    2. time complexity
    3. style and best practices

    You should be evaluating only and not attempting to solve the task.

    Only output "PASS" if all criteria are met and you have no further suggestions for improvements.

    Provide detailed feedback if there are areas that need improvement. You should specify what needs improvement and why.

    Only output JSON.
    `;

  const messages = [
    new SystemMessage(EVALUATOR_PROMPT),
    new HumanMessage(`Task: ${task} \n\n Code: ${code}`),
  ];
  
  return await model.withStructuredOutput(evaluatorSchema).invoke(messages);
}

// Main workflow function using a while loop instead of recursion
async function optimizeCode(task: string): Promise<string> {
  const MAX_ITERATIONS = 3;
  let currentIteration = 0;
  let currentCode = "";
  let currentFeedback: string | undefined;

  while (currentIteration < MAX_ITERATIONS) {
    // Generate code based on current feedback
    const generatedResult = await generateCode(task, currentFeedback, currentCode);
    currentCode = generatedResult.code;
    
    // Evaluate the generated code
    const evaluation = await evaluateCode(task, currentCode);
    
    // Check if code passes all criteria
    if (evaluation.pass === "PASS") {
      return currentCode;
    }

    // Update feedback and continue loop
    currentFeedback = evaluation.feedback;
    currentIteration++;
    
    // Log progress
    console.log(`Iteration ${currentIteration}/${MAX_ITERATIONS}: ${evaluation.pass}`);
    console.log("Feedback:", currentFeedback);
  }

  console.log("Reached maximum iterations. Returning last generated code.");
  return currentCode;
}

async function main() {
  const task = "Implement a Stack with: 1. push(x) 2. pop() 3. getMin() All operations should be O(1). In typescript.";
  const result = await optimizeCode(task);
  console.log("Final result:", result);
}

main().catch(console.error);