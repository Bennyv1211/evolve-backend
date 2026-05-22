import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const aiPlanUsage = new Map();
const AI_PLAN_WINDOW_MS = 24 * 60 * 60 * 1000;
const AI_PLAN_DAILY_LIMIT = Number(process.env.AI_PLAN_DAILY_LIMIT || 20);
const AI_PLAN_MAX_OUTPUT_TOKENS = Number(
  process.env.AI_PLAN_MAX_OUTPUT_TOKENS || 700,
);

app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/", (_req, res) => {
  res.json({ ok: true });
});

function clientKeyFromRequest(req) {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.trim()) {
    return forwarded.split(",")[0].trim();
  }
  return req.ip || "unknown";
}

function consumeDailyLimit(key, limit) {
  const now = Date.now();
  const current = aiPlanUsage.get(key);
  if (!current || now - current.windowStartedAt >= AI_PLAN_WINDOW_MS) {
    aiPlanUsage.set(key, { count: 1, windowStartedAt: now });
    return true;
  }
  if (current.count >= limit) {
    return false;
  }
  current.count += 1;
  aiPlanUsage.set(key, current);
  return true;
}

app.post("/ai/plan", async (req, res) => {
  try {
    const systemPrompt = String(req.body?.systemPrompt ?? "").trim();
    const userPrompt = String(req.body?.userPrompt ?? "").trim();
    const requestedModel = String(req.body?.model ?? "").trim();
    const model = requestedModel === "gpt-4o-mini" ? requestedModel : "gpt-4o-mini";

    if (!systemPrompt || !userPrompt) {
      return res.status(400).json({ error: "Missing planning prompt" });
    }

    if (systemPrompt.length > 5000 || userPrompt.length > 14000) {
      return res.status(400).json({ error: "Planning prompt too large" });
    }

    const requesterKey = clientKeyFromRequest(req);
    const allowed = consumeDailyLimit(`ai-plan:${requesterKey}`, AI_PLAN_DAILY_LIMIT);
    if (!allowed) {
      return res.status(429).json({ error: "Daily AI planning limit reached" });
    }

    const response = await client.responses.create({
      model,
      max_output_tokens: AI_PLAN_MAX_OUTPUT_TOKENS,
      input: [
        {
          role: "system",
          content: [{ type: "input_text", text: systemPrompt }],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: userPrompt }],
        },
      ],
    });

    const text = String(response.output_text ?? "").trim();
    if (!text) {
      return res.status(502).json({ error: "No AI output returned" });
    }

    return res.json({ text });
  } catch (error) {
    console.error("AI planning error:", error);
    return res.status(500).json({
      error: "AI planning failed",
    });
  }
});

app.post("/tts", async (req, res) => {
  try {
    const text = String(req.body?.text ?? "").trim();
    const voice = String(req.body?.voice ?? "shimmer").trim();

    if (!text) {
      return res.status(400).json({ error: "Missing text" });
    }

    const speech = await client.audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice,
      input: text,
      format: "mp3",
    });

    const buffer = Buffer.from(await speech.arrayBuffer());

    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", buffer.length);
    res.send(buffer);
  } catch (error) {
    console.error("TTS error:", error);
    res.status(500).json({
      error: "TTS generation failed",
      details: error?.message ?? String(error),
    });
  }
});

const port = Number(process.env.PORT || 10000);

app.listen(port, "0.0.0.0", () => {
  console.log(`Voice server running on port ${port}`);
});
