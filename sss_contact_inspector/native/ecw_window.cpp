// Minimal read-only ECW window decoder for the local SSS inspector.
// Based on the blocking API documented by the ECW JPEG 2000 SDK examples.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NCSECWClient.h"
#include "NCSErrors.h"

namespace {

bool parse_uint(const char *text, UINT32 &value) {
    if (!text || !*text) return false;
    errno = 0;
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || !end || *end || parsed > std::numeric_limits<UINT32>::max()) return false;
    value = static_cast<UINT32>(parsed);
    return true;
}

int fail(const char *message, NCSError error = NCS_SUCCESS) {
    if (error == NCS_SUCCESS) {
        std::fprintf(stderr, "ecw_window: %s\n", message);
    } else {
        std::fprintf(stderr, "ecw_window: %s: %s\n", message, NCSGetErrorText(error));
    }
    return 1;
}

void print_info(NCSFileViewFileInfo *file_info) {
    std::printf(
        "{\"width\":%u,\"height\":%u,\"bands\":%u,"
        "\"cell_increment_x\":%.17g,\"cell_increment_y\":%.17g,"
        "\"origin_x\":%.17g,\"origin_y\":%.17g,"
        "\"datum\":\"%s\",\"projection\":\"%s\"}\n",
        file_info->nSizeX, file_info->nSizeY, file_info->nBands,
        file_info->fCellIncrementX, file_info->fCellIncrementY,
        file_info->fOriginX, file_info->fOriginY,
        file_info->szDatum ? file_info->szDatum : "",
        file_info->szProjection ? file_info->szProjection : ""
    );
    std::fflush(stdout);
}

int info(const char *path) {
    NCSFileView *view = nullptr;
    NCSError error = NCScbmOpenFileView(const_cast<char *>(path), &view, nullptr);
    if (error != NCS_SUCCESS) return fail("cannot open ECW", error);
    NCSFileViewFileInfo *file_info = nullptr;
    error = NCScbmGetViewFileInfo(view, &file_info);
    if (error != NCS_SUCCESS || !file_info) {
        NCScbmCloseFileViewEx(view, TRUE);
        return fail("cannot read ECW metadata", error);
    }
    print_info(file_info);
    NCScbmCloseFileViewEx(view, TRUE);
    return 0;
}

bool decode_window(
    NCSFileView *view, NCSFileViewFileInfo *file_info,
    UINT32 x0, UINT32 y0, UINT32 x1, UINT32 y1,
    UINT32 out_width, UINT32 out_height, UINT32 band,
    std::vector<UINT8> &output, std::string &error_text
) {
    if (x1 < x0 || y1 < y0 || out_width == 0 || out_height == 0 ||
        out_width > 4096 || out_height > 4096) {
        error_text = "invalid or excessive output window";
        return false;
    }
    if (x1 >= file_info->nSizeX || y1 >= file_info->nSizeY || band >= file_info->nBands) {
        error_text = "source window or band is outside the ECW";
        return false;
    }
    UINT32 bands[1] = {band};
    NCSError error = NCScbmSetFileView(view, 1, bands, x0, y0, x1, y1, out_width, out_height);
    if (error != NCS_SUCCESS) {
        error_text = std::string("cannot set ECW view: ") + NCSGetErrorText(error);
        return false;
    }
    std::vector<UINT8> row(out_width);
    UINT8 *rows[1] = {row.data()};
    char header[80];
    int header_size = std::snprintf(header, sizeof(header), "P5\n%u %u\n255\n", out_width, out_height);
    output.assign(header, header + header_size);
    output.reserve(output.size() + static_cast<size_t>(out_width) * out_height);
    for (UINT32 y = 0; y < out_height; ++y) {
        NCSEcwReadStatus status = NCScbmReadViewLineBIL(view, rows);
        if (status != NCSECW_READ_OK) {
            error_text = "ECW scanline read failed";
            return false;
        }
        output.insert(output.end(), row.begin(), row.end());
    }
    return true;
}

int window(int argc, char **argv) {
    if (argc != 10) return fail("usage: ecw_window window FILE X0 Y0 X1 Y1 OUT_WIDTH OUT_HEIGHT BAND");
    UINT32 values[7];
    for (int index = 0; index < 7; ++index) {
        if (!parse_uint(argv[index + 3], values[index])) return fail("window arguments must be unsigned integers");
    }
    NCSFileView *view = nullptr;
    NCSError error = NCScbmOpenFileView(argv[2], &view, nullptr);
    if (error != NCS_SUCCESS) return fail("cannot open ECW", error);
    NCSFileViewFileInfo *file_info = nullptr;
    error = NCScbmGetViewFileInfo(view, &file_info);
    if (error != NCS_SUCCESS || !file_info) {
        NCScbmCloseFileViewEx(view, TRUE);
        return fail("cannot read ECW metadata", error);
    }
    std::vector<UINT8> output;
    std::string error_text;
    bool ok = decode_window(view, file_info, values[0], values[1], values[2], values[3], values[4], values[5], values[6], output, error_text);
    if (ok && std::fwrite(output.data(), 1, output.size(), stdout) != output.size()) {
        ok = false;
        error_text = "stdout write failed";
    }
    NCScbmCloseFileViewEx(view, TRUE);
    return ok ? 0 : fail(error_text.c_str());
}

int serve(const char *path) {
    NCSFileView *view = nullptr;
    NCSError error = NCScbmOpenFileView(const_cast<char *>(path), &view, nullptr);
    if (error != NCS_SUCCESS) return fail("cannot open ECW", error);
    NCSFileViewFileInfo *file_info = nullptr;
    error = NCScbmGetViewFileInfo(view, &file_info);
    if (error != NCS_SUCCESS || !file_info) {
        NCScbmCloseFileViewEx(view, TRUE);
        return fail("cannot read ECW metadata", error);
    }
    print_info(file_info);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string request_id;
        UINT32 x0, y0, x1, y1, out_width, out_height, band;
        if (!(input >> request_id >> x0 >> y0 >> x1 >> y1 >> out_width >> out_height >> band)) {
            std::printf("ERR ? malformed request\n");
            std::fflush(stdout);
            continue;
        }
        std::vector<UINT8> output;
        std::string error_text;
        if (!decode_window(view, file_info, x0, y0, x1, y1, out_width, out_height, band, output, error_text)) {
            std::printf("ERR %s %s\n", request_id.c_str(), error_text.c_str());
            std::fflush(stdout);
            continue;
        }
        std::printf("OK %s %zu\n", request_id.c_str(), output.size());
        std::fwrite(output.data(), 1, output.size(), stdout);
        std::fflush(stdout);
    }
    NCScbmCloseFileViewEx(view, TRUE);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) return fail("usage: ecw_window info FILE | serve FILE | window ...");
    NCSecwInit();
    NCSecwSetConfig(NCSCFG_CACHE_MAXMEM, static_cast<UINT32>(128U * 1024U * 1024U));
    NCSecwSetConfig(NCSCFG_CACHE_MAXOPEN, static_cast<UINT32>(2));
    NCSecwSetConfig(NCSCFG_PURGE_DELAY_MS, static_cast<NCSTimeStampMs>(0));
    NCSecwSetConfig(NCSCFG_FILE_PURGE_DELAY_MS, static_cast<NCSTimeStampMs>(0));
    NCSecwSetConfig(NCSCFG_MIN_FILE_PURGE_DELAY_MS, static_cast<NCSTimeStampMs>(0));
    int result = 1;
    if (std::strcmp(argv[1], "info") == 0 && argc == 3) result = info(argv[2]);
    else if (std::strcmp(argv[1], "serve") == 0 && argc == 3) result = serve(argv[2]);
    else if (std::strcmp(argv[1], "window") == 0) result = window(argc, argv);
    else result = fail("unknown mode");
    NCSecwShutdown();
    return result;
}
