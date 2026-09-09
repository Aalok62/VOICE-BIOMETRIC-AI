const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { execFile } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// File upload configuration using Multer
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname) || ".wav";
    cb(null, `audio-${uniqueSuffix}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB limit
});

// Health Endpoint
app.get("/api/health", (req, res) => {
  res.json({ status: "OK", service: "BioSound AI Backend" });
});

// Classification Endpoint
app.post("/api/classify", upload.single("audio"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file provided" });
  }

  const audioPath = req.file.path;
  const scriptPath = path.join(__dirname, "predict.py");

  // Call python inference script
  execFile("python", [scriptPath, audioPath], { maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
    // Cleanup temporary upload file
    fs.unlink(audioPath, () => {});

    if (error && !stdout.trim()) {
      console.error("Execution error:", stderr || error.message);
      return res.status(500).json({ error: "Failed to process audio classifier." });
    }

    try {
      const result = JSON.parse(stdout.trim());
      if (result.error) {
        return res.status(400).json({ error: result.error });
      }
      res.json(result);
    } catch (parseErr) {
      console.error("Parse error:", parseErr, "Raw Output:", stdout);
      res.status(500).json({ error: "Invalid JSON response from classifier." });
    }
  });
});

app.listen(PORT, () => {
  console.log(`🧠 BioSound AI Express server running on http://localhost:${PORT}`);
});
