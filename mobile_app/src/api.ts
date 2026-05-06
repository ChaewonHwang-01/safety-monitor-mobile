export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export type Detection = {
  class: string;
  confidence: number;
  status: "safe" | "risk" | string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  headwear_check?: string;
  headwear_confidence?: number;
};

export type FeedbackItem = {
  id: string;
  class: string;
  label: string;
  confidence: number;
  status: "safe" | "risk" | string;
  position: string;
  bbox: number[];
  reason: string;
};

export type RiskDetail = {
  risk_id: string;
  class: string;
  label: string;
  confidence: number;
  position: string;
  bbox: number[];
  reason: string;
  recommended_action: string;
};

export type VideoRiskEvent = {
  risk_id: string;
  class: string;
  raw_class?: string;
  label?: string;
  reason?: string;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  frame_count: number;
  detection_count: number;
  peak_count: number;
  severity: string;
};

export type AnalyzeResponse = {
  kind: "image" | "video";
  message: string;
  alert_count: number;
  total_detections: number;
  detections?: Detection[];
  risk_details?: RiskDetail[];
  feedback_items?: FeedbackItem[];
  annotated_image_base64?: string;
  source_image_base64?: string;
  preview_gif_base64?: string;
  risk_events?: VideoRiskEvent[];
  risk_frame_count?: number;
  raw_alert_count?: number;
};

export type DashboardResponse = {
  total_logs: number;
  total_risks: number;
  by_risk_type: Array<{ risk_type: string; label: string; description: string; count: number }>;
  date_range: { start_date?: string; end_date?: string };
  definitions: {
    total_logs: string;
    total_risks: string;
  };
};

function mediaTypeFor(uri: string, fallback: "image/jpeg" | "video/mp4") {
  const lower = uri.toLowerCase();
  if (lower.endsWith(".mov")) return "video/quicktime";
  if (lower.endsWith(".mp4")) return "video/mp4";
  if (lower.endsWith(".png")) return "image/png";
  return fallback;
}

async function analyzeMedia(
  baseUrl: string,
  uri: string,
  confidence: number,
  endpoint: "/analyze/image" | "/analyze/video",
): Promise<AnalyzeResponse> {
  const isVideo = endpoint.includes("video");
  const formData = new FormData();
  formData.append("confidence", String(confidence));
  if (isVideo) {
    formData.append("frame_stride", "3");
    formData.append("max_frames", "180");
  } else {
    formData.append("source_type", "mobile");
  }
  formData.append("file", {
    uri,
    name: isVideo ? "mobile_video.mp4" : "mobile_upload.jpg",
    type: mediaTypeFor(uri, isVideo ? "video/mp4" : "image/jpeg"),
  } as unknown as Blob);

  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    body: formData,
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(isVideo ? "동영상 분석에 실패했습니다." : "이미지 분석에 실패했습니다.");
  }
  return response.json();
}

export function analyzeImage(baseUrl: string, uri: string, confidence: number) {
  return analyzeMedia(baseUrl, uri, confidence, "/analyze/image");
}

export function analyzeVideo(baseUrl: string, uri: string, confidence: number) {
  return analyzeMedia(baseUrl, uri, confidence, "/analyze/video");
}

export async function fetchDashboard(
  baseUrl: string,
  startDate?: string,
  endDate?: string,
): Promise<DashboardResponse> {
  const params = new URLSearchParams();
  if (startDate) params.append("start_date", startDate);
  if (endDate) params.append("end_date", endDate);
  const query = params.toString();
  const response = await fetch(`${baseUrl}/events${query ? `?${query}` : ""}`);
  if (!response.ok) {
    throw new Error("대시보드를 불러오지 못했습니다.");
  }
  return response.json();
}

export async function saveFeedback(
  baseUrl: string,
  sourceImageBase64: string,
  bbox: number[],
  correctLabel: "helmet" | "cap_hat" | "bare_head",
): Promise<void> {
  const response = await fetch(`${baseUrl}/feedback/headwear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: sourceImageBase64,
      bbox,
      correct_label: correctLabel,
    }),
  });

  if (!response.ok) {
    throw new Error("피드백 저장에 실패했습니다.");
  }
}
