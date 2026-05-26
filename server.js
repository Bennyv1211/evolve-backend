import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const aiPlanUsage = new Map();
const tomtomUsage = new Map();
const AI_PLAN_WINDOW_MS = 24 * 60 * 60 * 1000;
const AI_PLAN_DAILY_LIMIT = Number(process.env.AI_PLAN_DAILY_LIMIT || 20);
const AI_PLAN_MAX_OUTPUT_TOKENS = Number(
  process.env.AI_PLAN_MAX_OUTPUT_TOKENS || 700,
);
const TOMTOM_API_KEY = String(process.env.TOMTOM_API_KEY || "").trim();
const TOMTOM_WINDOW_MS = 24 * 60 * 60 * 1000;
const TOMTOM_DAILY_LIMIT = Number(process.env.TOMTOM_DAILY_LIMIT || 250);

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

function consumeTomTomLimit(key, limit) {
  const now = Date.now();
  const current = tomtomUsage.get(key);
  if (!current || now - current.windowStartedAt >= TOMTOM_WINDOW_MS) {
    tomtomUsage.set(key, { count: 1, windowStartedAt: now });
    return true;
  }
  if (current.count >= limit) {
    return false;
  }
  current.count += 1;
  tomtomUsage.set(key, current);
  return true;
}

function tomtomRequesterKey(req) {
  return `tomtom:${clientKeyFromRequest(req)}`;
}

async function fetchTomTomJson(pathname, params) {
  if (!TOMTOM_API_KEY) {
    throw new Error("Missing TOMTOM_API_KEY");
  }

  const url = new URL(`https://api.tomtom.com${pathname}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  url.searchParams.set("key", TOMTOM_API_KEY);

  const response = await fetch(url, {
    headers: {
      Accept: "application/json",
      "User-Agent": "PulseTrip-Backend/1.0",
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`TomTom request failed (${response.status}): ${body}`);
  }

  return response.json();
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

app.post("/places/reverse", async (req, res) => {
  try {
    const latitude = Number(req.body?.latitude);
    const longitude = Number(req.body?.longitude);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      return res.status(400).json({ error: "Missing coordinates" });
    }

    const allowed = consumeTomTomLimit(tomtomRequesterKey(req), TOMTOM_DAILY_LIMIT);
    if (!allowed) {
      return res.status(429).json({ error: "Daily TomTom limit reached" });
    }

    const decoded = await fetchTomTomJson(
      `/search/2/reverseGeocode/${latitude.toFixed(6)},${longitude.toFixed(6)}.json`,
      {
        radius: 2000,
        language: "en-GB",
      },
    );

    const results = Array.isArray(decoded.addresses) ? decoded.addresses : [];
    if (results.length === 0) {
      return res.json({ city: null });
    }

    const address = results[0]?.address ?? {};
    const countryCode = String(address.countryCodeISO3 || "").trim().toUpperCase();
    let city = "";
    if (countryCode === "CYM") {
      city = "Grand Cayman";
    } else {
      city =
        String(address.municipality || "").trim() ||
        String(address.municipalitySubdivision || "").trim() ||
        String(address.countrySecondarySubdivision || "").trim() ||
        String(address.countrySubdivision || "").trim();
    }

    return res.json({ city: city || null, raw: results[0] ?? null });
  } catch (error) {
    console.error("TomTom reverse error:", error);
    return res.status(500).json({ error: "TomTom reverse geocoding failed" });
  }
});

app.post("/places/geocode", async (req, res) => {
  try {
    const city = String(req.body?.city ?? "").trim();
    if (!city) {
      return res.status(400).json({ error: "Missing city" });
    }

    const allowed = consumeTomTomLimit(tomtomRequesterKey(req), TOMTOM_DAILY_LIMIT);
    if (!allowed) {
      return res.status(429).json({ error: "Daily TomTom limit reached" });
    }

    const decoded = await fetchTomTomJson(`/search/2/geocode/${city}.json`, {
      limit: 1,
      language: "en-GB",
    });

    const results = Array.isArray(decoded.results) ? decoded.results : [];
    const position = results[0]?.position ?? null;
    if (!position) {
      return res.json({ lat: null, lon: null });
    }

    return res.json({
      lat: position.lat ?? null,
      lon: position.lon ?? null,
      raw: results[0] ?? null,
    });
  } catch (error) {
    console.error("TomTom geocode error:", error);
    return res.status(500).json({ error: "TomTom geocoding failed" });
  }
});

app.post("/places/nearby", async (req, res) => {
  try {
    const latitude = Number(req.body?.latitude);
    const longitude = Number(req.body?.longitude);
    const radiusKm = Number(req.body?.radiusKm ?? 40.2336);
    const limit = Number(req.body?.limit ?? 20);

    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      return res.status(400).json({ error: "Missing coordinates" });
    }

    const allowed = consumeTomTomLimit(tomtomRequesterKey(req), TOMTOM_DAILY_LIMIT);
    if (!allowed) {
      return res.status(429).json({ error: "Daily TomTom limit reached" });
    }

    const decoded = await fetchTomTomJson("/search/2/nearbySearch/.json", {
      lat: latitude.toFixed(6),
      lon: longitude.toFixed(6),
      radius: Math.round(radiusKm * 1000),
      limit: Math.max(1, Math.min(limit, 30)),
      idxSet: "POI",
      language: "en-GB",
      openingHours: "nextSevenDays",
      relatedPois: "off",
      timeZone: "iana",
    });

    return res.json({
      results: Array.isArray(decoded.results) ? decoded.results : [],
    });
  } catch (error) {
    console.error("TomTom nearby error:", error);
    return res.status(500).json({ error: "TomTom nearby search failed" });
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
