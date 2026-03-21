#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <fcitx-utils/event.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/standardpath.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx-config/configuration.h>
#include <fcitx-config/iniparser.h>
#include <fcitx-config/option.h>
#include <fcitx-config/rawconfig.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char *kDefaultHost = "127.0.0.1";
constexpr int kDefaultPort = 8787;
constexpr const char *kVoiceTypeConfigPath = "conf/voicetype.conf";

struct VoiceTypeSettings {
  fcitx::KeyList triggerKeys{fcitx::Key(FcitxKey_Alt_R)};
  fcitx::KeyList toggleKeys{fcitx::Key("Alt+z")};
  std::string host = kDefaultHost;
  int port = kDefaultPort;
};

fcitx::ListConstrain<fcitx::KeyConstrain> TriggerKeyListConstrain() {
  return fcitx::KeyListConstrain(fcitx::KeyConstrainFlags{
      fcitx::KeyConstrainFlag::AllowModifierOnly,
      fcitx::KeyConstrainFlag::AllowModifierLess,
  });
}

class VoiceTypeConfig : public fcitx::Configuration {
public:
  explicit VoiceTypeConfig(const VoiceTypeSettings &settings)
      : triggerKeys(this, "TriggerKey", "Trigger Key", settings.triggerKeys,
                    TriggerKeyListConstrain(), {},
                    fcitx::ToolTipAnnotation(
                        "Press and hold this key to record. Release to transcribe.")),
        toggleKeys(this, "ToggleKey", "Toggle Recording Key", settings.toggleKeys,
                   TriggerKeyListConstrain(), {},
                   fcitx::ToolTipAnnotation(
                       "Press once to start recording, press again to stop and transcribe. Default: Alt+Z.")),
        host(this, "Host", "ASR Host", settings.host, {},
             {}, fcitx::ToolTipAnnotation("ASR service host. 需要与UI配置完全相同。")),
        port(this, "Port", "ASR Port", settings.port, fcitx::IntConstrain(1, 65535),
             {}, fcitx::ToolTipAnnotation("ASR service port. 需要与UI配置完全相同。")) {}

  const char *typeName() const override { return "VoiceTypeConfig"; }

  VoiceTypeSettings settings() const {
    VoiceTypeSettings s;
    s.triggerKeys = triggerKeys.value();
    s.toggleKeys = toggleKeys.value();
    s.host = host.value();
    s.port = port.value();
    return s;
  }

  fcitx::Option<fcitx::KeyList, fcitx::ListConstrain<fcitx::KeyConstrain>,
                fcitx::DefaultMarshaller<fcitx::KeyList>, fcitx::ToolTipAnnotation>
      triggerKeys;

  fcitx::Option<fcitx::KeyList, fcitx::ListConstrain<fcitx::KeyConstrain>,
                fcitx::DefaultMarshaller<fcitx::KeyList>, fcitx::ToolTipAnnotation>
      toggleKeys;

  fcitx::Option<std::string, fcitx::NoConstrain<std::string>,
                fcitx::DefaultMarshaller<std::string>, fcitx::ToolTipAnnotation>
      host;

  fcitx::Option<int, fcitx::IntConstrain, fcitx::DefaultMarshaller<int>,
                fcitx::ToolTipAnnotation>
      port;
};

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

VoiceTypeSettings LoadVoiceTypeSettings() {
  VoiceTypeConfig config(VoiceTypeSettings{});
  fcitx::readAsIni(config, fcitx::StandardPath::Type::PkgConfig, kVoiceTypeConfigPath);
  return config.settings();
}

bool SaveVoiceTypeSettings(const VoiceTypeSettings &settings) {
  VoiceTypeConfig config(settings);
  return fcitx::safeSaveAsIni(config, fcitx::StandardPath::Type::PkgConfig,
                              kVoiceTypeConfigPath);
}

fcitx::Key ResolvePrimaryKey(const fcitx::KeyList &keys, const fcitx::Key &fallback) {
  if (!keys.empty()) {
    return keys.front().normalize();
  }
  return fallback;
}

std::string BuildAsrUrl(const std::string &baseUrl, const char *path) {
  return baseUrl + path;
}

} // namespace

class VoiceTypeAddon : public fcitx::AddonInstance {
public:
  enum class VisualState { Idle, RecordingVisual, TranscribingVisual };

  explicit VoiceTypeAddon(fcitx::Instance *instance) : instance_(instance) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    reloadConfig();

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

  void reloadConfig() override {
    settings_ = LoadVoiceTypeSettings();
    applySettings();
  }

  void save() override { SaveVoiceTypeSettings(settings_); }

  const fcitx::Configuration *getConfig() const override {
    uiConfig_ = std::make_unique<VoiceTypeConfig>(settings_);
    return uiConfig_.get();
  }

  void setConfig(const fcitx::RawConfig &rawConfig) override {
    auto config = std::make_unique<VoiceTypeConfig>(settings_);
    config->load(rawConfig, true);
    settings_ = config->settings();
    applySettings();
    SaveVoiceTypeSettings(settings_);
  }

private:
  bool isTriggerPressedEvent(const fcitx::Key &key) const {
    if (!triggerKey_.isModifier()) {
      return key.check(triggerKey_);
    }
    return key.sym() == triggerKey_.sym();
  }

  bool isTriggerReleasedEvent(const fcitx::Key &key) const {
    if (!triggerKey_.isModifier()) {
      return key.check(triggerKey_);
    }
    return key.isReleaseOfModifier(triggerKey_);
  }

  bool isToggleEvent(const fcitx::Key &key) const {
    if (toggleKey_.isModifier()) {
      return key.sym() == toggleKey_.sym();
    }
    return key.sym() == toggleKey_.sym() &&
           (key.states() & toggleKey_.states()) == toggleKey_.states();
  }

  void applySettings() {
    triggerKey_ = ResolvePrimaryKey(settings_.triggerKeys, fcitx::Key(FcitxKey_Alt_R));
    toggleKey_ = ResolvePrimaryKey(settings_.toggleKeys, fcitx::Key("Alt+z"));
    const auto &host = settings_.host.empty() ? std::string(kDefaultHost) : settings_.host;
    asrBaseUrl_ = std::string("http://") + host + ":" + std::to_string(settings_.port);
  }

  void handleKeyEvent(fcitx::Event &event) {
    auto &keyEvent = static_cast<fcitx::KeyEvent &>(event);

    // For modifier-only trigger keys (Alt/Ctrl), if user presses another key
    // while trigger is held, treat it as a shortcut chord and discard this
    // recording so we don't accidentally transcribe Alt+X/Ctrl+X patterns.
    if (triggerHeld_ && recording_ && triggerKey_.isModifier() &&
        !keyEvent.isRelease() && !isTriggerPressedEvent(keyEvent.key()) &&
        !isToggleEvent(keyEvent.key())) {
      stopAndDiscard(keyEvent.inputContext());
      triggerHeld_ = false;
      return;
    }

    if (isToggleEvent(keyEvent.key())) {
      if (keyEvent.isRelease()) {
        togglePressed_ = false;
        keyEvent.filterAndAccept();
        return;
      }
      if (togglePressed_) {
        keyEvent.filterAndAccept();
        return;
      }
      togglePressed_ = true;
      if (!recording_) {
        recordingByToggle_ = true;
        startRecording(keyEvent.inputContext());
      } else {
        stopAndCommit(keyEvent.inputContext());
      }
      keyEvent.filterAndAccept();
      return;
    }

    if (!keyEvent.isRelease()) {
      if (!isTriggerPressedEvent(keyEvent.key())) {
        return;
      }
      if (!triggerHeld_) {
        triggerHeld_ = true;
        if (!recording_) {
          recordingByToggle_ = false;
          startRecording(keyEvent.inputContext());
        }
      }
      keyEvent.filterAndAccept();
      return;
    }

    if (!isTriggerReleasedEvent(keyEvent.key())) {
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
    if (!PostJson(BuildAsrUrl(asrBaseUrl_, "/v1/recording/start"), "{}",
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
    if (!PostJson(BuildAsrUrl(asrBaseUrl_, "/v1/recording/stop"),
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
    recordingByToggle_ = false;
    activeIc_ = nullptr;
  }

  void stopAndDiscard(fcitx::InputContext *fallbackIc) {
    auto *ic = activeIc_ ? activeIc_ : fallbackIc;
    visualState_ = VisualState::TranscribingVisual;
    renderVisualFrame(ic);

    std::string response;
    std::string error;
    (void)PostJson(BuildAsrUrl(asrBaseUrl_, "/v1/recording/stop"),
                   R"({"language":"zh"})", &response, &error);

    if (ic) {
      clearPreedit(ic);
    }
    visualState_ = VisualState::Idle;
    recording_ = false;
    recordingByToggle_ = false;
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
      updatePreedit(ic, recordingByToggle_ ? "录音中...再按一次结束" : "录音中...");
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
  VoiceTypeSettings settings_;
  fcitx::Key triggerKey_ = fcitx::Key(FcitxKey_Alt_R);
  fcitx::Key toggleKey_ = fcitx::Key("Alt+z");
  bool togglePressed_ = false;
  mutable std::unique_ptr<VoiceTypeConfig> uiConfig_;
  std::string asrBaseUrl_ = "http://127.0.0.1:8787";
  bool recording_ = false;
  bool recordingByToggle_ = false;
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
