import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/", (_req, res) => {
  res.json({ ok: true });
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