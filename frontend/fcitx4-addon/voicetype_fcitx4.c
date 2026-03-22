#include <ctype.h>
#include <fcitx/fcitx.h>
#include <fcitx/frontend.h>
#include <fcitx/hook.h>
#include <fcitx/ime.h>
#include <fcitx/instance.h>
#include <fcitx/module.h>
#include <fcitx/ui.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/keysym.h>
#include <fcitx-utils/utils.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *ptr;
    size_t len;
} Buffer;

typedef struct {
    FcitxInstance *instance;
    FcitxInputContext *active_ic;
    int recording;
    int record_mode; /* 0 idle, 1 hold, 2 toggle */
    int hold_pressed;
    int hold_blocked;
    int toggle_latched;
    char host[128];
    int port;
    int hold_is_right_alt;
    int hold_is_left_alt;
    FcitxHotkey hold_hotkey;
    char hold_name[32];
    FcitxHotkey toggle_hotkey;
    uint64_t watchdog_timeout_id;
} VoiceTypeFcitx4;

static const char *kDefaultHost = "127.0.0.1";
static const int kDefaultPort = 8787;
static const char *kDefaultHoldKey = "ALT_R";
static const char *kDefaultHoldModifier = "DISABLED";
static const char *kDefaultToggleKey = "ALT_z";
static const int kRecordModeIdle = 0;
static const int kRecordModeHold = 1;
static const int kRecordModeToggle = 2;
static const long kRecordingWatchdogMs = 120000;

static void trim_ws(char *s) {
    char *end = NULL;
    while (*s && isspace((unsigned char)*s)) {
        memmove(s, s + 1, strlen(s));
    }
    if (!*s) {
        return;
    }
    end = s + strlen(s) - 1;
    while (end >= s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
}

static void strip_quotes(char *s) {
    size_t n = strlen(s);
    if (n >= 2 && ((s[0] == '"' && s[n - 1] == '"') || (s[0] == '\'' && s[n - 1] == '\''))) {
        memmove(s, s + 1, n - 2);
        s[n - 2] = '\0';
    }
}

static int set_hotkey_with_modifier_alias(FcitxHotkey *hotkey, const char *key) {
    if (!key || !*key) {
        return 0;
    }
    if (strcasecmp(key, "RALT") == 0 || strcasecmp(key, "ALT_R") == 0 ||
        strcasecmp(key, "RIGHT_ALT") == 0) {
        hotkey->sym = FcitxKey_Alt_R;
        hotkey->state = FcitxKeyState_None;
        hotkey->desc = strdup("RALT");
        return 1;
    }
    if (strcasecmp(key, "LALT") == 0 || strcasecmp(key, "ALT_L") == 0 ||
        strcasecmp(key, "LEFT_ALT") == 0) {
        hotkey->sym = FcitxKey_Alt_L;
        hotkey->state = FcitxKeyState_None;
        hotkey->desc = strdup("LALT");
        return 1;
    }
    return 0;
}

static void set_hold_key(VoiceTypeFcitx4 *self, const char *key) {
    const char *k = key ? key : "";
    self->hold_is_right_alt = 0;
    self->hold_is_left_alt = 0;
    memset(&self->hold_hotkey, 0, sizeof(self->hold_hotkey));

    if (strcasecmp(k, "RALT") == 0 || strcasecmp(k, "ALT_R") == 0 || strcasecmp(k, "RIGHT_ALT") == 0) {
        self->hold_is_right_alt = 1;
        self->hold_hotkey.sym = FcitxKey_Alt_R;
        self->hold_hotkey.state = FcitxKeyState_None;
        self->hold_hotkey.desc = strdup("ALT_R");
        snprintf(self->hold_name, sizeof(self->hold_name), "ALT_R");
        return;
    }
    if (strcasecmp(k, "LALT") == 0 || strcasecmp(k, "ALT_L") == 0 || strcasecmp(k, "LEFT_ALT") == 0) {
        self->hold_is_left_alt = 1;
        self->hold_hotkey.sym = FcitxKey_Alt_L;
        self->hold_hotkey.state = FcitxKeyState_None;
        self->hold_hotkey.desc = strdup("ALT_L");
        snprintf(self->hold_name, sizeof(self->hold_name), "ALT_L");
        return;
    }

    if (!set_hotkey_with_modifier_alias(&self->hold_hotkey, k)) {
        FcitxHotkeySetKey(k, &self->hold_hotkey);
    }
    snprintf(
        self->hold_name,
        sizeof(self->hold_name),
        "%s",
        self->hold_hotkey.desc ? self->hold_hotkey.desc : kDefaultHoldKey
    );
}

static int is_disabled_modifier(const char *value) {
    if (!value || !*value) {
        return 1;
    }
    return strcasecmp(value, "DISABLED") == 0 || strcasecmp(value, "NONE") == 0 ||
           strcasecmp(value, "OFF") == 0;
}

static int match_hold_key(const VoiceTypeFcitx4 *self, FcitxKeySym sym, unsigned int state) {
    if (self->hold_is_right_alt) {
        return sym == FcitxKey_Alt_R || sym == FcitxKey_ISO_Level3_Shift || sym == FcitxKey_Mode_switch ||
               sym == FcitxKey_Meta_R;
    }
    if (self->hold_is_left_alt) {
        return sym == FcitxKey_Alt_L || sym == FcitxKey_Meta_L;
    }
    return FcitxHotkeyIsHotKey(sym, state, &self->hold_hotkey);
}

static int load_runtime_config(VoiceTypeFcitx4 *self) {
    char path[PATH_MAX];
    char line[512];
    FILE *fp = NULL;
    const char *home = getenv("HOME");
    int in_section = 0;
    char hold_key[64];
    char hold_modifier[32];
    char toggle_key[64];

    snprintf(self->host, sizeof(self->host), "%s", kDefaultHost);
    self->port = kDefaultPort;
    memset(&self->toggle_hotkey, 0, sizeof(self->toggle_hotkey));
    snprintf(hold_key, sizeof(hold_key), "%s", kDefaultHoldKey);
    snprintf(hold_modifier, sizeof(hold_modifier), "%s", kDefaultHoldModifier);
    snprintf(toggle_key, sizeof(toggle_key), "%s", kDefaultToggleKey);

    if (home && *home) {
        snprintf(path, sizeof(path), "%s/.config/fcitx/conf/fcitx-voicetype.config", home);
        fp = fopen(path, "r");
    }
    if (!fp) {
        fp = fopen("/usr/share/fcitx/conf/fcitx-voicetype.config", "r");
    }
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            char *eq = NULL;
            char *key = NULL;
            char *value = NULL;
            trim_ws(line);
            if (line[0] == '\0' || line[0] == '#') {
                continue;
            }
            if (line[0] == '[') {
                in_section = (strncmp(line, "[VoiceType]", 11) == 0);
                continue;
            }
            if (!in_section) {
                continue;
            }
            eq = strchr(line, '=');
            if (!eq) {
                continue;
            }
            *eq = '\0';
            key = line;
            value = eq + 1;
            trim_ws(key);
            trim_ws(value);
            strip_quotes(value);
            if (strcmp(key, "Host") == 0 && value[0]) {
                snprintf(self->host, sizeof(self->host), "%s", value);
                continue;
            }
            if (strcmp(key, "Port") == 0 && value[0]) {
                int p = atoi(value);
                if (p > 0 && p <= 65535) {
                    self->port = p;
                }
                continue;
            }
            if (strcmp(key, "HoldKey") == 0 && value[0]) {
                snprintf(hold_key, sizeof(hold_key), "%s", value);
                continue;
            }
            if (strcmp(key, "HoldModifier") == 0 && value[0]) {
                snprintf(hold_modifier, sizeof(hold_modifier), "%s", value);
                continue;
            }
            if (strcmp(key, "ToggleKey") == 0 && value[0]) {
                snprintf(toggle_key, sizeof(toggle_key), "%s", value);
                continue;
            }
        }
        fclose(fp);
    }

    if (!is_disabled_modifier(hold_modifier)) {
        set_hold_key(self, hold_modifier);
    } else {
        set_hold_key(self, hold_key);
    }
    FcitxHotkeySetKey(toggle_key, &self->toggle_hotkey);
    return 0;
}

static int send_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, buf + off, len - off, 0);
        if (n <= 0) {
            return -1;
        }
        off += (size_t)n;
    }
    return 0;
}

static int http_post_json(const char *host, int port, const char *path, const char *json_body, Buffer *out) {
    struct addrinfo hints;
    struct addrinfo *res = NULL;
    struct addrinfo *rp = NULL;
    char port_str[16];
    char header[1024];
    int fd = -1;
    int ret = -1;
    int status_code = 0;
    char tmp[4096];
    char *header_end = NULL;

    out->ptr = NULL;
    out->len = 0;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        return -1;
    }

    for (rp = res; rp != NULL; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    if (!rp || fd < 0) {
        goto cleanup;
    }

    snprintf(header, sizeof(header),
             "POST %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Content-Type: application/json\r\n"
             "Connection: close\r\n"
             "Content-Length: %zu\r\n\r\n",
             path,
             host,
             port,
             strlen(json_body));
    if (send_all(fd, header, strlen(header)) != 0 || send_all(fd, json_body, strlen(json_body)) != 0) {
        goto cleanup;
    }

    out->ptr = (char *)malloc(1);
    if (!out->ptr) {
        goto cleanup;
    }
    out->ptr[0] = '\0';
    out->len = 0;

    while (1) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        char *next = NULL;
        if (n < 0) {
            goto cleanup;
        }
        if (n == 0) {
            break;
        }
        next = (char *)realloc(out->ptr, out->len + (size_t)n + 1);
        if (!next) {
            goto cleanup;
        }
        out->ptr = next;
        memcpy(out->ptr + out->len, tmp, (size_t)n);
        out->len += (size_t)n;
        out->ptr[out->len] = '\0';
    }

    if (!out->ptr) {
        goto cleanup;
    }
    if (sscanf(out->ptr, "HTTP/%*s %d", &status_code) != 1) {
        goto cleanup;
    }
    if (status_code < 200 || status_code >= 300) {
        goto cleanup;
    }
    header_end = strstr(out->ptr, "\r\n\r\n");
    if (header_end) {
        size_t body_len = out->len - (size_t)(header_end + 4 - out->ptr);
        memmove(out->ptr, header_end + 4, body_len);
        out->ptr[body_len] = '\0';
        out->len = body_len;
    }
    ret = 0;

cleanup:
    if (ret != 0) {
        free(out->ptr);
        out->ptr = NULL;
        out->len = 0;
    }
    if (fd >= 0) {
        close(fd);
    }
    if (res) {
        freeaddrinfo(res);
    }
    return ret;
}

static char *extract_text_field(const char *json) {
    const char *p = strstr(json, "\"text\"");
    const char *q = NULL;
    const char *start = NULL;
    char *out = NULL;
    size_t i = 0;
    size_t o = 0;

    if (!p) {
        return NULL;
    }
    p = strchr(p, ':');
    if (!p) {
        return NULL;
    }
    p++;
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    if (*p != '"') {
        return NULL;
    }
    start = ++p;
    q = start;
    while (*q) {
        if (*q == '"' && q > start && q[-1] != '\\') {
            break;
        }
        q++;
    }
    if (!*q || q <= start) {
        return NULL;
    }
    out = (char *)malloc((size_t)(q - start) + 1);
    if (!out) {
        return NULL;
    }

    for (i = 0; start + i < q; i++) {
        char c = start[i];
        if (c == '\\' && (start + i + 1) < q) {
            char n = start[i + 1];
            if (n == '"' || n == '\\' || n == '/') {
                out[o++] = n;
                i++;
                continue;
            }
            if (n == 'n') {
                out[o++] = '\n';
                i++;
                continue;
            }
            if (n == 't') {
                out[o++] = '\t';
                i++;
                continue;
            }
            if (n == 'r') {
                out[o++] = '\r';
                i++;
                continue;
            }
        }
        out[o++] = c;
    }
    out[o] = '\0';
    return out;
}

static int asr_start(VoiceTypeFcitx4 *self) {
    Buffer out;
    if (http_post_json(self->host, self->port, "/v1/recording/start", "{\"sample_rate\":16000}", &out) != 0) {
        FcitxLog(ERROR, "voicetype-fcitx4: start request failed (%s:%d)", self->host, self->port);
        return -1;
    }
    free(out.ptr);
    self->recording = 1;
    self->hold_blocked = 0;
    return 0;
}

static void show_aux_message(VoiceTypeFcitx4 *self, const char *text) {
    FcitxInputState *input = FcitxInstanceGetInputState(self->instance);
    FcitxMessages *aux_up = NULL;
    if (!input || !text) {
        return;
    }
    aux_up = FcitxInputStateGetAuxUp(input);
    if (!aux_up) {
        return;
    }
    FcitxMessagesSetMessageCount(aux_up, 0);
    FcitxMessagesAddMessageAtLast(aux_up, MSG_TIPS, "%s", text);
    FcitxMessagesSetMessageChanged(aux_up, true);
    FcitxUIUpdateInputWindow(self->instance);
}

static void clear_aux_message(VoiceTypeFcitx4 *self) {
    FcitxInstanceCleanInputWindowUp(self->instance);
    FcitxUIUpdateInputWindow(self->instance);
}

static void commit_text(VoiceTypeFcitx4 *self, const char *text) {
    FcitxInputContext *ic = self->active_ic ? self->active_ic : FcitxInstanceGetCurrentIC(self->instance);
    if (!ic || !text || !*text) {
        return;
    }
    FcitxInstanceCommitString(self->instance, ic, text);
}

static int asr_stop_and_commit(VoiceTypeFcitx4 *self, int inject) {
    Buffer out;
    char *text = NULL;
    /* Always exit recording state on stop action, even if backend request fails. */
    self->recording = 0;
    self->record_mode = kRecordModeIdle;
    self->hold_pressed = 0;
    self->hold_blocked = 0;
    clear_aux_message(self);
    if (self->watchdog_timeout_id &&
        FcitxInstanceCheckTimeoutById(self->instance, self->watchdog_timeout_id)) {
        FcitxInstanceRemoveTimeoutById(self->instance, self->watchdog_timeout_id);
    }
    self->watchdog_timeout_id = 0;

    if (http_post_json(self->host, self->port, "/v1/recording/stop", "{}", &out) != 0) {
        FcitxLog(ERROR, "voicetype-fcitx4: stop request failed (%s:%d)", self->host, self->port);
        return -1;
    }
    if (inject) {
        text = extract_text_field(out.ptr);
        if (text && *text) {
            commit_text(self, text);
        }
    }
    free(text);
    free(out.ptr);
    return 0;
}

static void watchdog_timeout_cb(void *arg) {
    VoiceTypeFcitx4 *self = (VoiceTypeFcitx4 *)arg;
    if (!self || !self->recording) {
        return;
    }
    FcitxLog(WARNING, "voicetype-fcitx4: watchdog timeout reached, force stop");
    show_aux_message(self, "识别中...");
    (void)asr_stop_and_commit(self, 0);
    self->active_ic = NULL;
}

static void arm_watchdog(VoiceTypeFcitx4 *self) {
    if (self->watchdog_timeout_id &&
        FcitxInstanceCheckTimeoutById(self->instance, self->watchdog_timeout_id)) {
        FcitxInstanceRemoveTimeoutById(self->instance, self->watchdog_timeout_id);
    }
    self->watchdog_timeout_id =
        FcitxInstanceAddTimeout(self->instance, kRecordingWatchdogMs, watchdog_timeout_cb, self);
}

static void on_input_unfocus(void *arg) {
    VoiceTypeFcitx4 *self = (VoiceTypeFcitx4 *)arg;
    if (!self || !self->recording) {
        return;
    }
    FcitxLog(INFO, "voicetype-fcitx4: input unfocus, force stop recording");
    show_aux_message(self, "识别中...");
    (void)asr_stop_and_commit(self, 0);
    self->active_ic = NULL;
}

static int is_modifier_sym(FcitxKeySym sym) {
    return sym == FcitxKey_Alt_L || sym == FcitxKey_Alt_R || sym == FcitxKey_Control_L ||
           sym == FcitxKey_Control_R || sym == FcitxKey_Shift_L || sym == FcitxKey_Shift_R ||
           sym == FcitxKey_Super_L || sym == FcitxKey_Super_R;
}

static int match_hotkey(FcitxKeySym sym, unsigned int state, const FcitxHotkey *hotkey) {
    if (FcitxHotkeyIsHotKey(sym, state, hotkey)) {
        return 1;
    }
    if (is_modifier_sym(hotkey->sym) && sym == hotkey->sym) {
        return 1;
    }
    return 0;
}

static boolean press_filter(void *arg, FcitxKeySym sym, unsigned int state, INPUT_RETURN_VALUE *retval) {
    VoiceTypeFcitx4 *self = (VoiceTypeFcitx4 *)arg;
    FcitxKeySym nsym = sym;
    unsigned int nstate = state;
    FcitxHotkeyGetKey(sym, state, &nsym, &nstate);

    if (match_hold_key(self, nsym, nstate)) {
        self->hold_pressed = 1;
        if (!self->recording && self->record_mode == kRecordModeIdle) {
            self->active_ic = FcitxInstanceGetCurrentIC(self->instance);
            if (asr_start(self) == 0) {
                self->record_mode = kRecordModeHold;
                show_aux_message(self, "录音中...");
                arm_watchdog(self);
            }
        }
        /* Do not consume hold-key press event. Otherwise some fcitx4 paths
         * may skip matching release callbacks, causing recording to be stuck. */
        return false;
    }

    if (match_hotkey(nsym, nstate, &self->toggle_hotkey)) {
        if (!self->toggle_latched) {
            self->toggle_latched = 1;
            if (!self->recording && self->record_mode == kRecordModeIdle) {
                self->active_ic = FcitxInstanceGetCurrentIC(self->instance);
                if (asr_start(self) == 0) {
                    self->record_mode = kRecordModeToggle;
                    show_aux_message(self, "录音中...再按一次结束");
                    arm_watchdog(self);
                }
            } else if (self->record_mode == kRecordModeToggle) {
                show_aux_message(self, "识别中...");
                (void)asr_stop_and_commit(self, 1);
            }
        }
        *retval = IRV_DO_NOTHING;
        return true;
    }

    if (self->recording && self->record_mode == kRecordModeHold && self->hold_pressed &&
        !match_hold_key(self, nsym, nstate)) {
        /* Modifier-only hold key mixed with another key chord: stop immediately
         * to avoid "stuck recording" when release event is swallowed upstream. */
        self->hold_blocked = 1;
        show_aux_message(self, "识别中...");
        (void)asr_stop_and_commit(self, 0);
        self->active_ic = NULL;
        *retval = IRV_DO_NOTHING;
        return true;
    }

    return false;
}

static boolean release_filter(void *arg, FcitxKeySym sym, unsigned int state, INPUT_RETURN_VALUE *retval) {
    VoiceTypeFcitx4 *self = (VoiceTypeFcitx4 *)arg;
    FcitxKeySym nsym = sym;
    unsigned int nstate = state;
    FCITX_UNUSED(nstate);
    FcitxHotkeyGetKey(sym, state, &nsym, &nstate);

    if (match_hold_key(self, nsym, nstate)) {
        if (self->recording && self->record_mode == kRecordModeHold && self->hold_pressed) {
            show_aux_message(self, "识别中...");
            (void)asr_stop_and_commit(self, self->hold_blocked ? 0 : 1);
            *retval = IRV_DO_NOTHING;
            self->active_ic = NULL;
        }
        self->hold_pressed = 0;
        return true;
    }

    if (match_hotkey(nsym, nstate, &self->toggle_hotkey)) {
        self->toggle_latched = 0;
    }

    return false;
}

static void *voicetype_create(FcitxInstance *instance) {
    VoiceTypeFcitx4 *self = fcitx_utils_new(VoiceTypeFcitx4);
    if (!self) {
        return NULL;
    }
    memset(self, 0, sizeof(*self));
    self->instance = instance;
    self->record_mode = kRecordModeIdle;
    load_runtime_config(self);

    FcitxInstanceRegisterPreInputFilter(instance, (FcitxKeyFilterHook){press_filter, self});
    FcitxInstanceRegisterPreReleaseInputFilter(instance, (FcitxKeyFilterHook){release_filter, self});
    FcitxInstanceRegisterInputUnFocusHook(instance, (FcitxIMEventHook){on_input_unfocus, self});
    FcitxLog(INFO, "voicetype-fcitx4: loaded (hold=%s, toggle=%s, asr=%s:%d)",
             self->hold_name[0] ? self->hold_name : kDefaultHoldKey,
             self->toggle_hotkey.desc ? self->toggle_hotkey.desc : kDefaultToggleKey,
             self->host,
             self->port);
    return self;
}

static void voicetype_destroy(void *arg) {
    VoiceTypeFcitx4 *self = (VoiceTypeFcitx4 *)arg;
    if (!self) {
        return;
    }
    if (self->watchdog_timeout_id &&
        FcitxInstanceCheckTimeoutById(self->instance, self->watchdog_timeout_id)) {
        FcitxInstanceRemoveTimeoutById(self->instance, self->watchdog_timeout_id);
    }
    FcitxHotkeyFree(&self->hold_hotkey);
    FcitxHotkeyFree(&self->toggle_hotkey);
    free(self);
}

static void voicetype_reload(void *arg) {
    VoiceTypeFcitx4 *self = (VoiceTypeFcitx4 *)arg;
    if (!self) {
        return;
    }
    FcitxHotkeyFree(&self->hold_hotkey);
    FcitxHotkeyFree(&self->toggle_hotkey);
    load_runtime_config(self);
    if (!self->recording) {
        self->record_mode = kRecordModeIdle;
        self->hold_pressed = 0;
        self->hold_blocked = 0;
        self->toggle_latched = 0;
    }
    FcitxLog(INFO, "voicetype-fcitx4: config reloaded (asr=%s:%d)", self->host, self->port);
}

FCITX_DEFINE_PLUGIN(fcitx_voicetype, module, FcitxModule) = {
    .Create = voicetype_create,
    .SetFD = NULL,
    .ProcessEvent = NULL,
    .Destroy = voicetype_destroy,
    .ReloadConfig = voicetype_reload,
};
