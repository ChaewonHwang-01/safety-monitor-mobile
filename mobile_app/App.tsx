import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import { StatusBar } from "expo-status-bar";
import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import {
  AnalyzeResponse,
  DashboardResponse,
  DEFAULT_API_BASE_URL,
  FeedbackItem,
  RiskDetail,
  analyzeImage,
  analyzeVideo,
  fetchDashboard,
  saveFeedback,
} from "./src/api";

type Tab = "scan" | "dashboard" | "feedback";
type MediaKind = "image" | "video";
type FeedbackLabel = "helmet" | "cap_hat" | "bare_head";

const labelOptions: Array<{ label: string; value: FeedbackLabel }> = [
  { label: "안전모", value: "helmet" },
  { label: "일반 캡", value: "cap_hat" },
  { label: "안전모 미착용", value: "bare_head" },
];

const classHelp: Record<string, string> = {
  no_helmet: "머리에 아무것도 착용하지 않은 상태입니다.",
  cap_hat: "머리에 착용물이 있지만 안전모가 아닌 상태입니다.",
  helmet: "안전 보호구로 인정된 탐지입니다.",
};

const weekDays = ["일", "월", "화", "수", "목", "금", "토"];

function toDateKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function addDays(date: Date, amount: number) {
  const next = new Date(date);
  next.setDate(next.getDate() + amount);
  return next;
}

function monthTitle(date: Date) {
  return `${date.getFullYear()}년 ${date.getMonth() + 1}월`;
}

function parseDateKey(value?: string | null) {
  if (!value) return null;
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) return null;
  return new Date(year, month - 1, day);
}

function isDateInRange(dateKey: string, startDate: string, endDate: string) {
  return dateKey >= startDate && dateKey <= endDate;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("scan");
  const [selectedUri, setSelectedUri] = useState<string | null>(null);
  const [selectedKind, setSelectedKind] = useState<MediaKind>("image");
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [selectedFeedbackId, setSelectedFeedbackId] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<FeedbackLabel>("cap_hat");
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [confidence, setConfidence] = useState(0.35);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [dashboardStartDate, setDashboardStartDate] = useState(() => toDateKey(addDays(new Date(), -6)));
  const [dashboardEndDate, setDashboardEndDate] = useState(() => toDateKey(new Date()));

  const feedbackItems = analysis?.feedback_items ?? [];
  const selectedFeedbackItem = useMemo(() => {
    if (!feedbackItems.length) return null;
    return feedbackItems.find((item) => item.id === selectedFeedbackId) ?? feedbackItems[0];
  }, [feedbackItems, selectedFeedbackId]);

  useEffect(() => {
    if (tab === "dashboard") {
      refreshDashboard();
    }
  }, [tab, dashboardStartDate, dashboardEndDate]);

  function updateConfidence(delta: number) {
    setConfidence((current) => Math.max(0.1, Math.min(0.9, Number((current + delta).toFixed(2)))));
  }

  async function pickImage(fromCamera: boolean) {
    const permission = fromCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("권한 필요", "이미지를 선택하거나 촬영하려면 권한이 필요합니다.");
      return;
    }

    const result = fromCamera
      ? await ImagePicker.launchCameraAsync({ quality: 1 })
      : await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          quality: 1,
        });

    if (result.canceled || !result.assets[0]?.uri) return;
    setSelectedUri(result.assets[0].uri);
    setSelectedKind("image");
    setAnalysis(null);
  }

  async function pickVideo() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("권한 필요", "동영상을 선택하려면 갤러리 권한이 필요합니다.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 0.8,
    });

    if (result.canceled || !result.assets[0]?.uri) return;
    setSelectedUri(result.assets[0].uri);
    setSelectedKind("video");
    setAnalysis(null);
  }

  async function runAnalysis() {
    if (!selectedUri) {
      Alert.alert("입력 필요", "먼저 현장 사진 또는 영상을 선택하세요.");
      return;
    }

    setLoading(true);
    try {
      const result =
        selectedKind === "video"
          ? await analyzeVideo(apiBaseUrl.trim(), selectedUri, confidence)
          : await analyzeImage(apiBaseUrl.trim(), selectedUri, confidence);
      setAnalysis(result);
      setSelectedFeedbackId(result.feedback_items?.[0]?.id ?? null);
      if (result.alert_count > 0) {
        if (voiceEnabled) {
          Speech.stop();
          Speech.speak(result.message, { language: "ko-KR", rate: 0.95 });
        }
        Alert.alert("안전 경고", result.message);
      }
    } catch (error) {
      Alert.alert("분석 실패", error instanceof Error ? error.message : "서버 연결을 확인하세요.");
    } finally {
      setLoading(false);
    }
  }

  async function refreshDashboard() {
    setDashboardLoading(true);
    try {
      setDashboard(await fetchDashboard(apiBaseUrl.trim(), dashboardStartDate, dashboardEndDate));
    } catch (error) {
      Alert.alert("대시보드 오류", error instanceof Error ? error.message : "서버 연결을 확인하세요.");
    } finally {
      setDashboardLoading(false);
    }
  }

  async function submitFeedback() {
    if (!analysis?.source_image_base64 || !selectedFeedbackItem) {
      Alert.alert("피드백 대상 없음", "이미지 분석 결과에서 수정할 탐지 항목을 선택하세요.");
      return;
    }

    try {
      await saveFeedback(apiBaseUrl.trim(), analysis.source_image_base64, selectedFeedbackItem.bbox, selectedLabel);
      Alert.alert("저장 완료", "수정 샘플이 다음 보정 학습 데이터로 저장되었습니다.");
    } catch (error) {
      Alert.alert("저장 실패", error instanceof Error ? error.message : "서버 연결을 확인하세요.");
    }
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.title}>작업자 안전 모니터</Text>
        <Text style={styles.subtitle}>PPE 탐지 · 경고 · 피드백 학습 데이터</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <ServerPanel apiBaseUrl={apiBaseUrl} setApiBaseUrl={setApiBaseUrl} />

        {tab === "scan" && (
          <View>
            <View style={styles.heroPanel}>
              <Text style={styles.sectionTitle}>현장 입력 분석</Text>
              <Text style={styles.bodyText}>
                사진, 갤러리 이미지, 저장된 동영상을 분석합니다. 영상은 여러 프레임의 경고를 시간 기반 이벤트로 묶어 요약합니다.
              </Text>
              <View style={styles.actionRow}>
                <IconButton icon="camera" label="촬영" onPress={() => pickImage(true)} />
                <IconButton icon="images" label="사진" onPress={() => pickImage(false)} />
                <IconButton icon="film" label="영상" onPress={pickVideo} />
              </View>
            </View>

            <View style={styles.controlPanel}>
              <View style={styles.controlHeader}>
                <Text style={styles.labelTitle}>탐지 신뢰도 기준</Text>
                <Text style={styles.confidenceValue}>{confidence.toFixed(2)}</Text>
              </View>
              <Text style={styles.helpText}>
                낮추면 더 많이 잡고, 높이면 더 확실한 탐지만 남깁니다. 현재 모델 테스트 때 쓰던 기준을 앱에서도 조정할 수 있습니다.
              </Text>
              <View style={styles.stepperRow}>
                <Pressable style={styles.stepperButton} onPress={() => updateConfidence(-0.05)}>
                  <Text style={styles.stepperText}>-</Text>
                </Pressable>
                <View style={styles.stepperBar}>
                  <View style={[styles.stepperFill, { width: `${((confidence - 0.1) / 0.8) * 100}%` }]} />
                </View>
                <Pressable style={styles.stepperButton} onPress={() => updateConfidence(0.05)}>
                  <Text style={styles.stepperText}>+</Text>
                </Pressable>
              </View>
              <Pressable style={styles.toggleRow} onPress={() => setVoiceEnabled((value) => !value)}>
                <Ionicons name={voiceEnabled ? "volume-high" : "volume-mute"} size={20} color="#ef4444" />
                <Text style={styles.toggleText}>{voiceEnabled ? "음성 경고 켜짐" : "음성 경고 꺼짐"}</Text>
              </Pressable>
            </View>

            {selectedUri && (
              selectedKind === "video" ? (
                <View style={styles.selectedVideoBox}>
                  <Ionicons name="film" size={34} color="#ef4444" />
                  <Text style={styles.bodyText}>동영상이 선택되었습니다. 분석에는 시간이 걸릴 수 있습니다.</Text>
                </View>
              ) : (
                <Image source={{ uri: selectedUri }} style={styles.previewImage} />
              )
            )}

            <Pressable style={[styles.primaryButton, loading && styles.disabledButton]} onPress={runAnalysis} disabled={loading}>
              {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryButtonText}>분석 시작</Text>}
            </Pressable>

            {analysis && <AnalysisResult analysis={analysis} onSelectFeedback={setSelectedFeedbackId} />}
          </View>
        )}

        {tab === "dashboard" && (
          <DashboardView
            dashboard={dashboard}
            loading={dashboardLoading}
            startDate={dashboardStartDate}
            endDate={dashboardEndDate}
            setStartDate={setDashboardStartDate}
            setEndDate={setDashboardEndDate}
            onRefresh={refreshDashboard}
          />
        )}

        {tab === "feedback" && (
          <FeedbackView
            analysis={analysis}
            feedbackItems={feedbackItems}
            selectedFeedbackItem={selectedFeedbackItem}
            selectedFeedbackId={selectedFeedbackId}
            setSelectedFeedbackId={setSelectedFeedbackId}
            selectedLabel={selectedLabel}
            setSelectedLabel={setSelectedLabel}
            submitFeedback={submitFeedback}
          />
        )}
      </ScrollView>

      <View style={styles.bottomNav}>
        <NavItem icon="scan" label="분석" active={tab === "scan"} onPress={() => setTab("scan")} />
        <NavItem icon="stats-chart" label="대시보드" active={tab === "dashboard"} onPress={() => setTab("dashboard")} />
        <NavItem icon="create" label="피드백" active={tab === "feedback"} onPress={() => setTab("feedback")} />
      </View>
    </SafeAreaView>
  );
}

function ServerPanel({ apiBaseUrl, setApiBaseUrl }: { apiBaseUrl: string; setApiBaseUrl: (value: string) => void }) {
  return (
    <View style={styles.serverPanel}>
      <Text style={styles.serverLabel}>분석 서버 주소</Text>
      <TextInput
        value={apiBaseUrl}
        onChangeText={setApiBaseUrl}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="http://노트북IP:8000"
        style={styles.serverInput}
      />
      <Text style={styles.serverHint}>실제 휴대폰에서는 노트북의 Wi-Fi IP 주소를 넣어야 합니다.</Text>
    </View>
  );
}

function AnalysisResult({
  analysis,
  onSelectFeedback,
}: {
  analysis: AnalyzeResponse;
  onSelectFeedback: (id: string) => void;
}) {
  const riskDetails = analysis.risk_details ?? [];

  return (
    <View style={styles.resultPanel}>
      <View style={styles.metricRow}>
        <Metric label={analysis.kind === "video" ? "위험 이벤트" : "위험"} value={analysis.alert_count} tone={analysis.alert_count ? "danger" : "safe"} />
        <Metric label={analysis.kind === "video" ? "처리 프레임" : "탐지"} value={analysis.total_detections} tone="neutral" />
      </View>
      <Text style={[styles.alertText, analysis.alert_count ? styles.dangerText : styles.safeText]}>{analysis.message}</Text>

      {analysis.kind === "video" ? (
        <View>
          {analysis.preview_gif_base64 ? (
            <Image source={{ uri: `data:image/gif;base64,${analysis.preview_gif_base64}` }} style={styles.resultImage} />
          ) : (
            <Text style={styles.bodyText}>동영상 미리보기가 없습니다.</Text>
          )}
          <Text style={styles.helpText}>
            프레임 단위 누적 감지 {analysis.raw_alert_count ?? 0}건을 시간 기반 위험 이벤트 {analysis.alert_count}건으로 묶었습니다.
          </Text>
          {(analysis.risk_events ?? []).map((event) => (
            <View style={styles.riskCard} key={event.risk_id}>
              <View style={styles.riskHeader}>
                <Text style={styles.riskId}>{event.risk_id}</Text>
                <Text style={styles.riskLabel}>{event.label ?? event.class}</Text>
              </View>
              <Text style={styles.bodyText}>
                {event.start_time}s - {event.end_time}s · {event.severity}
              </Text>
              {!!event.raw_class && event.raw_class !== event.class && (
                <Text style={styles.logMeta}>원본 판정: {event.raw_class}</Text>
              )}
            </View>
          ))}
        </View>
      ) : (
        <View>
          <Image source={{ uri: `data:image/jpeg;base64,${analysis.annotated_image_base64}` }} style={styles.resultImage} />
          <Text style={styles.sectionTitle}>위험 이벤트 상세</Text>
          {riskDetails.length === 0 ? (
            <Text style={styles.bodyText}>위험 이벤트가 없습니다.</Text>
          ) : (
            riskDetails.map((risk) => (
              <RiskCard key={risk.risk_id} risk={risk} onPress={() => onSelectFeedback(`D${risk.risk_id.slice(1)}`)} />
            ))
          )}
        </View>
      )}
    </View>
  );
}

function DashboardView({
  dashboard,
  loading,
  startDate,
  endDate,
  setStartDate,
  setEndDate,
  onRefresh,
}: {
  dashboard: DashboardResponse | null;
  loading: boolean;
  startDate: string;
  endDate: string;
  setStartDate: (value: string) => void;
  setEndDate: (value: string) => void;
  onRefresh: () => void;
}) {
  return (
    <View>
      <View style={styles.sectionHeaderRow}>
        <Text style={styles.sectionTitle}>누적 위험 대시보드</Text>
        <Pressable onPress={onRefresh}>
          <Ionicons name="refresh" size={22} color="#ef4444" />
        </Pressable>
      </View>
      <CalendarRangePicker
        startDate={startDate}
        endDate={endDate}
        setStartDate={setStartDate}
        setEndDate={setEndDate}
      />
      {loading ? (
        <ActivityIndicator color="#ef4444" />
      ) : dashboard ? (
        <View>
          <Text style={styles.selectedRangeText}>
            조회 기간: {startDate} ~ {endDate}
          </Text>
          <View style={styles.metricRow}>
            <Metric label="위험 분석 기록" value={dashboard.total_logs} tone="neutral" />
            <Metric label="위험 이벤트" value={dashboard.total_risks} tone="danger" />
          </View>
          <Text style={styles.helpText}>
            위험 분석 기록: {dashboard.definitions.total_logs}{"\n"}
            위험 이벤트: {dashboard.definitions.total_risks}
          </Text>
          <SummaryList title="미착용 유형별" items={dashboard.by_risk_type.map((item) => [item.label, item.count, item.description])} />
          <View style={styles.summaryPanel}>
            <Text style={styles.sectionTitle}>판정 용어</Text>
            {Object.entries(classHelp).map(([key, value]) => (
              <Text style={styles.helpText} key={key}>
                {key}: {value}
              </Text>
            ))}
          </View>
        </View>
      ) : (
        <Text style={styles.bodyText}>대시보드를 불러오세요.</Text>
      )}
    </View>
  );
}

function CalendarRangePicker({
  startDate,
  endDate,
  setStartDate,
  setEndDate,
}: {
  startDate: string;
  endDate: string;
  setStartDate: (value: string) => void;
  setEndDate: (value: string) => void;
}) {
  const initialMonth = parseDateKey(endDate) ?? new Date();
  const [visibleMonth, setVisibleMonth] = useState(new Date(initialMonth.getFullYear(), initialMonth.getMonth(), 1));
  const [isSelectingEnd, setIsSelectingEnd] = useState(false);
  const firstDay = new Date(visibleMonth.getFullYear(), visibleMonth.getMonth(), 1);
  const daysInMonth = new Date(visibleMonth.getFullYear(), visibleMonth.getMonth() + 1, 0).getDate();
  const leadingBlanks = firstDay.getDay();
  const dayCells = [
    ...Array.from({ length: leadingBlanks }, (_, index) => ({ key: `blank-${index}`, dateKey: "", day: "" })),
    ...Array.from({ length: daysInMonth }, (_, index) => {
      const day = index + 1;
      const dateKey = toDateKey(new Date(visibleMonth.getFullYear(), visibleMonth.getMonth(), day));
      return { key: dateKey, dateKey, day: String(day) };
    }),
  ];

  function moveMonth(amount: number) {
    setVisibleMonth((current) => new Date(current.getFullYear(), current.getMonth() + amount, 1));
  }

  function selectDate(dateKey: string) {
    if (!dateKey) return;
    if (!isSelectingEnd) {
      setStartDate(dateKey);
      setEndDate(dateKey);
      setIsSelectingEnd(true);
      return;
    }
    if (dateKey < startDate) {
      setEndDate(startDate);
      setStartDate(dateKey);
    } else {
      setEndDate(dateKey);
    }
    setIsSelectingEnd(false);
  }

  function setQuickRange(days: number) {
    const today = new Date();
    setEndDate(toDateKey(today));
    setStartDate(toDateKey(addDays(today, -(days - 1))));
    setIsSelectingEnd(false);
    setVisibleMonth(new Date(today.getFullYear(), today.getMonth(), 1));
  }

  function setAllRange() {
    setStartDate("2000-01-01");
    setEndDate(toDateKey(new Date()));
    setIsSelectingEnd(false);
    const today = new Date();
    setVisibleMonth(new Date(today.getFullYear(), today.getMonth(), 1));
  }

  return (
    <View style={styles.calendarPanel}>
      <View style={styles.calendarHeader}>
        <Pressable style={styles.calendarNavButton} onPress={() => moveMonth(-1)}>
          <Ionicons name="chevron-back" size={18} color="#374151" />
        </Pressable>
        <Text style={styles.calendarTitle}>{monthTitle(visibleMonth)}</Text>
        <Pressable style={styles.calendarNavButton} onPress={() => moveMonth(1)}>
          <Ionicons name="chevron-forward" size={18} color="#374151" />
        </Pressable>
      </View>
      <View style={styles.quickRangeRow}>
        <Pressable style={styles.quickRangeChip} onPress={() => setQuickRange(1)}>
          <Text style={styles.quickRangeText}>오늘</Text>
        </Pressable>
        <Pressable style={styles.quickRangeChip} onPress={() => setQuickRange(7)}>
          <Text style={styles.quickRangeText}>7일</Text>
        </Pressable>
        <Pressable style={styles.quickRangeChip} onPress={() => setQuickRange(30)}>
          <Text style={styles.quickRangeText}>30일</Text>
        </Pressable>
        <Pressable style={styles.quickRangeChip} onPress={setAllRange}>
          <Text style={styles.quickRangeText}>전체</Text>
        </Pressable>
      </View>
      <View style={styles.weekRow}>
        {weekDays.map((day) => (
          <Text style={styles.weekText} key={day}>{day}</Text>
        ))}
      </View>
      <View style={styles.dayGrid}>
        {dayCells.map((cell) => {
          const selected = Boolean(cell.dateKey && isDateInRange(cell.dateKey, startDate, endDate));
          const edge = cell.dateKey === startDate || cell.dateKey === endDate;
          return (
            <Pressable
              key={cell.key}
              style={[styles.dayCell, selected && styles.dayCellSelected, edge && styles.dayCellEdge]}
              onPress={() => selectDate(cell.dateKey)}
            >
              <Text style={[styles.dayText, selected && styles.dayTextSelected, edge && styles.dayTextEdge]}>
                {cell.day}
              </Text>
            </Pressable>
          );
        })}
      </View>
      <Text style={styles.helpText}>시작일을 누른 뒤 종료일을 누르면 해당 기간의 위험만 집계합니다.</Text>
    </View>
  );
}

function FeedbackView({
  analysis,
  feedbackItems,
  selectedFeedbackItem,
  selectedFeedbackId,
  setSelectedFeedbackId,
  selectedLabel,
  setSelectedLabel,
  submitFeedback,
}: {
  analysis: AnalyzeResponse | null;
  feedbackItems: FeedbackItem[];
  selectedFeedbackItem: FeedbackItem | null;
  selectedFeedbackId: string | null;
  setSelectedFeedbackId: (id: string) => void;
  selectedLabel: FeedbackLabel;
  setSelectedLabel: (label: FeedbackLabel) => void;
  submitFeedback: () => void;
}) {
  return (
    <View>
      <Text style={styles.sectionTitle}>오탐/미탐 피드백</Text>
      <Text style={styles.bodyText}>
        위험 박스뿐 아니라 안전모로 인식된 박스도 정정할 수 있습니다. 저장된 crop은 다음 보정 학습 데이터로 사용됩니다.
      </Text>
      {!analysis || analysis.kind !== "image" || feedbackItems.length === 0 ? (
        <View style={styles.emptyPanel}>
          <Text style={styles.bodyText}>이미지 분석 후 정정할 탐지 항목을 선택하세요. 동영상은 현재 이벤트 요약만 지원합니다.</Text>
        </View>
      ) : (
        <View style={styles.resultPanel}>
          <Text style={styles.labelTitle}>정정할 탐지 항목</Text>
          {feedbackItems.map((item) => (
            <Pressable
              key={item.id}
              style={[styles.feedbackItem, selectedFeedbackId === item.id && styles.feedbackItemSelected]}
              onPress={() => setSelectedFeedbackId(item.id)}
            >
              <Text style={styles.riskId}>{item.id}</Text>
              <Text style={styles.bodyText}>
                {item.label} · {item.status} · 신뢰도 {item.confidence}
              </Text>
            </Pressable>
          ))}

          {selectedFeedbackItem && (
            <View style={styles.selectedFeedbackBox}>
              <Text style={styles.labelTitle}>선택된 항목</Text>
              <Text style={styles.bodyText}>
                {selectedFeedbackItem.label} · {selectedFeedbackItem.position} · 신뢰도 {selectedFeedbackItem.confidence}
              </Text>
            </View>
          )}

          <Text style={styles.labelTitle}>올바른 라벨</Text>
          <View style={styles.chipRow}>
            {labelOptions.map((option) => (
              <Pressable
                key={option.value}
                style={[styles.chip, selectedLabel === option.value && styles.chipSelected]}
                onPress={() => setSelectedLabel(option.value)}
              >
                <Text style={[styles.chipText, selectedLabel === option.value && styles.chipTextSelected]}>
                  {option.label}
                </Text>
              </Pressable>
            ))}
          </View>
          <Pressable style={styles.primaryButton} onPress={submitFeedback}>
            <Text style={styles.primaryButtonText}>피드백 저장</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

function IconButton({ icon, label, onPress }: { icon: keyof typeof Ionicons.glyphMap; label: string; onPress: () => void }) {
  return (
    <Pressable style={styles.iconButton} onPress={onPress}>
      <Ionicons name={icon} size={22} color="#ef4444" />
      <Text style={styles.iconButtonText}>{label}</Text>
    </Pressable>
  );
}

function Metric({ label, value, tone }: { label: string; value: number; tone: "danger" | "safe" | "neutral" }) {
  return (
    <View style={styles.metricCard}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, tone === "danger" && styles.dangerText, tone === "safe" && styles.safeText]}>
        {value}
      </Text>
    </View>
  );
}

function RiskCard({ risk, onPress }: { risk: RiskDetail; onPress?: () => void }) {
  return (
    <Pressable style={styles.riskCard} onPress={onPress}>
      <View style={styles.riskHeader}>
        <Text style={styles.riskId}>{risk.risk_id}</Text>
        <Text style={styles.riskLabel}>{risk.label}</Text>
      </View>
      <Text style={styles.bodyText}>{risk.reason}</Text>
      <Text style={styles.logMeta}>
        {risk.position} · 신뢰도 {risk.confidence}
      </Text>
    </Pressable>
  );
}

function SummaryList({ title, items }: { title: string; items: Array<[string, number, string?]> }) {
  return (
    <View style={styles.summaryPanel}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {items.length === 0 ? (
        <Text style={styles.bodyText}>기록이 없습니다.</Text>
      ) : (
        items.map(([label, count, description]) => (
          <View style={styles.summaryRow} key={label}>
            <View style={styles.summaryTextBlock}>
              <Text style={styles.bodyText}>{label}</Text>
              {!!description && <Text style={styles.helpText}>{description}</Text>}
            </View>
            <Text style={styles.summaryCount}>{count}</Text>
          </View>
        ))
      )}
    </View>
  );
}

function NavItem({
  icon,
  label,
  active,
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable style={styles.navItem} onPress={onPress}>
      <Ionicons name={icon} size={22} color={active ? "#ef4444" : "#6b7280"} />
      <Text style={[styles.navText, active && styles.navTextActive]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#f8fafc" },
  header: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
    backgroundColor: "#ffffff",
    borderBottomColor: "#e5e7eb",
    borderBottomWidth: 1,
  },
  title: { fontSize: 24, fontWeight: "800", color: "#111827" },
  subtitle: { marginTop: 4, fontSize: 13, color: "#6b7280" },
  content: { padding: 18, paddingBottom: 120 },
  serverPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    marginBottom: 14,
  },
  serverLabel: { fontSize: 13, fontWeight: "800", color: "#374151", marginBottom: 8 },
  serverInput: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: "#111827",
    backgroundColor: "#f9fafb",
  },
  serverHint: { marginTop: 7, color: "#6b7280", fontSize: 12 },
  heroPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 18,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  sectionHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  sectionTitle: { fontSize: 18, fontWeight: "800", color: "#111827", marginBottom: 10 },
  bodyText: { fontSize: 14, lineHeight: 20, color: "#374151" },
  helpText: { fontSize: 12, lineHeight: 18, color: "#6b7280", marginTop: 4 },
  actionRow: { flexDirection: "row", gap: 8, marginTop: 16 },
  iconButton: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#fecaca",
    backgroundColor: "#fff1f2",
  },
  iconButtonText: { color: "#991b1b", fontWeight: "700" },
  controlPanel: {
    marginTop: 14,
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  controlHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  confidenceValue: { color: "#ef4444", fontWeight: "800", fontSize: 18 },
  stepperRow: { flexDirection: "row", alignItems: "center", gap: 10, marginTop: 12 },
  stepperButton: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: "#fee2e2",
    alignItems: "center",
    justifyContent: "center",
  },
  stepperText: { color: "#991b1b", fontWeight: "900", fontSize: 20 },
  stepperBar: {
    flex: 1,
    height: 10,
    borderRadius: 999,
    backgroundColor: "#e5e7eb",
    overflow: "hidden",
  },
  stepperFill: { height: "100%", borderRadius: 999, backgroundColor: "#ef4444" },
  toggleRow: { flexDirection: "row", alignItems: "center", gap: 8, marginTop: 14 },
  toggleText: { fontWeight: "700", color: "#374151" },
  previewImage: {
    marginTop: 16,
    width: "100%",
    height: 260,
    borderRadius: 10,
    backgroundColor: "#e5e7eb",
  },
  selectedVideoBox: {
    marginTop: 16,
    minHeight: 150,
    borderRadius: 10,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#e5e7eb",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
    gap: 10,
  },
  resultImage: {
    width: "100%",
    height: 300,
    borderRadius: 10,
    backgroundColor: "#e5e7eb",
    marginVertical: 14,
  },
  primaryButton: {
    marginTop: 16,
    borderRadius: 8,
    paddingVertical: 15,
    alignItems: "center",
    backgroundColor: "#ef4444",
  },
  primaryButtonText: { color: "#ffffff", fontWeight: "800", fontSize: 16 },
  disabledButton: { opacity: 0.7 },
  resultPanel: {
    marginTop: 18,
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 16,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  metricRow: { flexDirection: "row", gap: 10, marginBottom: 12 },
  metricCard: {
    flex: 1,
    backgroundColor: "#f9fafb",
    borderRadius: 8,
    padding: 14,
    borderColor: "#e5e7eb",
    borderWidth: 1,
  },
  metricLabel: { color: "#6b7280", fontSize: 13 },
  metricValue: { marginTop: 6, color: "#111827", fontSize: 28, fontWeight: "800" },
  alertText: { fontSize: 15, fontWeight: "700", marginBottom: 8 },
  dangerText: { color: "#dc2626" },
  safeText: { color: "#16a34a" },
  riskCard: {
    padding: 14,
    borderRadius: 8,
    backgroundColor: "#fff7ed",
    borderWidth: 1,
    borderColor: "#fed7aa",
    marginBottom: 10,
  },
  riskHeader: { flexDirection: "row", justifyContent: "space-between", marginBottom: 8 },
  riskId: { fontWeight: "800", color: "#9a3412" },
  riskLabel: { fontWeight: "800", color: "#dc2626" },
  logMeta: { marginTop: 6, fontSize: 12, color: "#6b7280" },
  summaryPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 16,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    marginBottom: 14,
  },
  summaryRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "#f3f4f6",
    paddingVertical: 10,
    gap: 12,
  },
  summaryTextBlock: { flex: 1 },
  summaryCount: { fontWeight: "800", color: "#ef4444" },
  selectedRangeText: {
    marginBottom: 10,
    color: "#374151",
    fontSize: 13,
    fontWeight: "700",
  },
  calendarPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    marginBottom: 14,
  },
  calendarHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 10,
  },
  calendarTitle: { fontWeight: "800", fontSize: 16, color: "#111827" },
  calendarNavButton: {
    width: 34,
    height: 34,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 8,
    backgroundColor: "#f3f4f6",
  },
  quickRangeRow: { flexDirection: "row", gap: 8, marginBottom: 12 },
  quickRangeChip: {
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: "#fff1f2",
    borderWidth: 1,
    borderColor: "#fecaca",
  },
  quickRangeText: { color: "#991b1b", fontWeight: "800", fontSize: 12 },
  weekRow: { flexDirection: "row", marginBottom: 5 },
  weekText: { flex: 1, textAlign: "center", color: "#6b7280", fontSize: 11, fontWeight: "800" },
  dayGrid: { flexDirection: "row", flexWrap: "wrap" },
  dayCell: {
    width: "14.2857%",
    aspectRatio: 1,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 8,
  },
  dayCellSelected: { backgroundColor: "#fee2e2" },
  dayCellEdge: { backgroundColor: "#ef4444" },
  dayText: { color: "#374151", fontSize: 12, fontWeight: "700" },
  dayTextSelected: { color: "#991b1b" },
  dayTextEdge: { color: "#ffffff" },
  compactLogRow: {
    borderTopWidth: 1,
    borderTopColor: "#f3f4f6",
    paddingVertical: 10,
  },
  compactLogTitle: { fontWeight: "700", color: "#111827", fontSize: 13 },
  emptyPanel: {
    backgroundColor: "#ffffff",
    borderRadius: 10,
    padding: 18,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    marginTop: 12,
  },
  labelTitle: { marginTop: 10, marginBottom: 8, fontSize: 14, fontWeight: "800", color: "#111827" },
  feedbackItem: {
    borderRadius: 8,
    padding: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    marginBottom: 8,
    backgroundColor: "#ffffff",
  },
  feedbackItemSelected: { borderColor: "#ef4444", backgroundColor: "#fff1f2" },
  selectedFeedbackBox: { marginTop: 8, padding: 12, backgroundColor: "#f9fafb", borderRadius: 8 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: "#d1d5db",
    backgroundColor: "#ffffff",
  },
  chipSelected: { borderColor: "#ef4444", backgroundColor: "#fee2e2" },
  chipText: { color: "#374151", fontWeight: "700" },
  chipTextSelected: { color: "#991b1b" },
  bottomNav: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "space-around",
    paddingTop: 10,
    paddingBottom: 18,
    backgroundColor: "#ffffff",
    borderTopWidth: 1,
    borderTopColor: "#e5e7eb",
  },
  navItem: { alignItems: "center", gap: 4, minWidth: 86 },
  navText: { fontSize: 12, color: "#6b7280", fontWeight: "700" },
  navTextActive: { color: "#ef4444" },
});
