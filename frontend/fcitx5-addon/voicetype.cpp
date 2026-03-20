#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <fcitx-utils/event.h>
#include <fcitx-utils/key.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char *kBaseUrl = "http://127.0.0.1:8787";

size_t CurlWriteCallback(void *contents, size_t size, size_t nmemb,
                         std::string *output) {
  const size_t total = size * nmemb;
  output->append(static_cast<char *>(contents), total);
  return total;
}

bool PostJson(const std::string &url, const std::string &payload,
              std::string *response, std::string *error) {
  CURL *curl = curl_easy_init();
  if (!curl) {
    if (error) {
      *error = "curl init failed";
    }
    return false;
  }

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");

  response->clear();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload.size());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 120000L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 2000L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);

  const auto res = curl_easy_perform(curl);
  long code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK) {
    if (error) {
      *error = curl_easy_strerror(res);
    }
    return false;
  }

  if (code >= 400) {
    if (error) {
      *error = "HTTP " + std::to_string(code);
    }
    return false;
  }

  return true;
}

} // namespace

class VoiceTypeAddon : public fcitx::AddonInstance {
public:
  enum class VisualState { Idle, RecordingVisual, TranscribingVisual };

  explicit VoiceTypeAddon(fcitx::Instance *instance) : instance_(instance) {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    eventHandlers_.emplace_back(instance_->watchEvent(
        fcitx::EventType::InputContextKeyEvent,
        fcitx::EventWatcherPhase::PreInputMethod,
        [this](fcitx::Event &event) { handleKeyEvent(event); }));

    eventHandlers_.emplace_back(instance_->watchEvent(
        fcitx::EventType::InputContextDestroyed,
        fcitx::EventWatcherPhase::PreInputMethod,
        [this](fcitx::Event &event) {
          auto &icEvent = static_cast<fcitx::InputContextEvent &>(event);
          if (activeIc_ == icEvent.inputContext()) {
            activeIc_ = nullptr;
            recording_ = false;
            visualState_ = VisualState::Idle;
          }
        }));

    constexpr uint64_t kUiTickUsec = 50000;
    uiTicker_ = instance_->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, fcitx::now(CLOCK_MONOTONIC) + kUiTickUsec,
        kUiTickUsec, [this](fcitx::EventSourceTime *, uint64_t) {
          onUiTick();
          return true;
        });
  }

  ~VoiceTypeAddon() override { curl_global_cleanup(); }

private:
  void handleKeyEvent(fcitx::Event &event) {
    auto &keyEvent = static_cast<fcitx::KeyEvent &>(event);

    if (!keyEvent.key().check(FcitxKey_Alt_R)) {
      return;
    }

    if (!keyEvent.isRelease()) {
      if (!triggerHeld_) {
        triggerHeld_ = true;
        if (!recording_) {
          startRecording(keyEvent.inputContext());
        }
      }
      keyEvent.filterAndAccept();
      return;
    }

    if (triggerHeld_ && recording_) {
      stopAndCommit(keyEvent.inputContext());
    }
    triggerHeld_ = false;
    keyEvent.filterAndAccept();
  }

  void startRecording(fcitx::InputContext *ic) {
    std::string response;
    std::string error;
    if (!PostJson(std::string(kBaseUrl) + "/v1/recording/start", "{}",
                  &response, &error)) {
      visualState_ = VisualState::Idle;
      updatePreedit(ic, "ASR start failed: " + error);
      return;
    }

    recording_ = true;
    activeIc_ = ic;
    visualState_ = VisualState::RecordingVisual;
    recordingFrameIndex_ = 0;
    renderVisualFrame(activeIc_);
  }

  void stopAndCommit(fcitx::InputContext *fallbackIc) {
    auto *ic = activeIc_ ? activeIc_ : fallbackIc;
    visualState_ = VisualState::TranscribingVisual;
    transcribingFrameIndex_ = 0;
    renderVisualFrame(ic);

    std::string response;
    std::string error;
    if (!PostJson(std::string(kBaseUrl) + "/v1/recording/stop",
                  R"({"language":"zh"})", &response, &error)) {
      visualState_ = VisualState::Idle;
      updatePreedit(ic, "ASR stop failed: " + error);
      recording_ = false;
      activeIc_ = nullptr;
      return;
    }

    std::string text;
    try {
      const auto j = nlohmann::json::parse(response);
      if (!j.value("success", false)) {
        const auto err = j.value("error", std::string("unknown"));
        visualState_ = VisualState::Idle;
        updatePreedit(ic, "ASR error: " + err);
        recording_ = false;
        activeIc_ = nullptr;
        return;
      }
      text = j.value("text", std::string{});
    } catch (...) {
      visualState_ = VisualState::Idle;
      updatePreedit(ic, "ASR parse response failed");
      recording_ = false;
      activeIc_ = nullptr;
      return;
    }

    if (ic) {
      clearPreedit(ic);
    }
    if (ic && !text.empty()) {
      ic->commitString(text);
    }

    visualState_ = VisualState::Idle;
    recording_ = false;
    activeIc_ = nullptr;
  }

  void onUiTick() {
    if (visualState_ == VisualState::Idle) {
      return;
    }
    if (!activeIc_) {
      return;
    }
    renderVisualFrame(activeIc_);
  }

  void renderVisualFrame(fcitx::InputContext *ic) {
    if (!ic) {
      return;
    }

    if (visualState_ == VisualState::RecordingVisual) {
      updatePreedit(ic, "说话中...");
      return;
    }

    if (visualState_ == VisualState::TranscribingVisual) {
      updatePreedit(ic, "识别中...");
    }
  }

  void updatePreedit(fcitx::InputContext *ic, const std::string &text) {
    if (!ic) {
      return;
    }
    // Keep hints inline in the client text field and avoid extra popup panel.
    fcitx::Text empty;
    ic->inputPanel().setPreedit(empty);
    fcitx::Text preedit;
    preedit.append(text);
    ic->inputPanel().setClientPreedit(preedit);
    ic->updatePreedit();
  }

  void clearPreedit(fcitx::InputContext *ic) {
    if (!ic) {
      return;
    }
    fcitx::Text empty;
    ic->inputPanel().setPreedit(empty);
    ic->inputPanel().setClientPreedit(empty);
    ic->updatePreedit();
  }

  fcitx::Instance *instance_;
  bool recording_ = false;
  bool triggerHeld_ = false;
  fcitx::InputContext *activeIc_ = nullptr;
  VisualState visualState_ = VisualState::Idle;
  size_t recordingFrameIndex_ = 0;
  size_t transcribingFrameIndex_ = 0;
  std::unique_ptr<fcitx::EventSourceTime> uiTicker_;
  std::vector<std::unique_ptr<fcitx::HandlerTableEntry<fcitx::EventHandler>>>
      eventHandlers_;
};

class VoiceTypeAddonFactory : public fcitx::AddonFactory {
  fcitx::AddonInstance *create(fcitx::AddonManager *manager) override {
    return new VoiceTypeAddon(manager->instance());
  }
};

#ifdef VOICETYPE_HAVE_ADDON_FACTORY_V2
FCITX_ADDON_FACTORY_V2(voicetype, VoiceTypeAddonFactory);
#else
FCITX_ADDON_FACTORY(VoiceTypeAddonFactory);
#endif
