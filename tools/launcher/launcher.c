/* PMAF Annotator launcher (Windows).
 *
 * Native replacement for START_HERE.bat: batch launchers that start bundled
 * executables trip Windows Defender heuristics; a plain console exe with
 * version metadata does not.  Behavior is identical to the old .bat:
 *   cd to the package folder -> run the bundled Python with serve.py
 *   (falling back to a system Python on PATH) -> keep the window open.
 *
 * Build (from macOS/Linux with mingw-w64): see build.sh next to this file.
 */
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define PATHBUF 32768

static void pause_exit(int code)
{
    fputws(L"\npress Enter to close this window...", stdout);
    fflush(stdout);
    getwchar();
    ExitProcess(code);
}

/* Ctrl+C is delivered to every process on this console.  Let Python receive
 * it and shut the server down cleanly; the launcher survives so it can show
 * the exit message instead of the window vanishing mid-shutdown. */
static BOOL WINAPI ctrl_handler(DWORD ev)
{
    return ev == CTRL_C_EVENT || ev == CTRL_BREAK_EVENT;
}

int wmain(void)
{
    static wchar_t exedir[PATHBUF];
    static wchar_t python[PATHBUF];
    static wchar_t cmd[PATHBUF + 64];

    SetConsoleTitleW(L"PMAF Annotator");

    DWORD n = GetModuleFileNameW(NULL, exedir, PATHBUF);
    if (n == 0 || n >= PATHBUF - 1) {
        fputws(L"ERROR: cannot determine the launcher's own location.\n", stdout);
        pause_exit(1);
    }
    wchar_t *slash = wcsrchr(exedir, L'\\');
    if (slash)
        *slash = L'\0';
    if (!SetCurrentDirectoryW(exedir)) {
        wprintf(L"ERROR: cannot enter the package folder:\n  %ls\n", exedir);
        pause_exit(1);
    }

    fputws(L"PMAF Annotator - starting up\n", stdout);

    /* Pick a Python: bundled runtime first, then anything on PATH. */
    const wchar_t *bundled = L"runtime\\windows\\python.exe";
    if (GetFileAttributesW(bundled) != INVALID_FILE_ATTRIBUTES) {
        wcscpy(python, bundled);
    } else if (!SearchPathW(NULL, L"python", L".exe", PATHBUF, python, NULL)) {
        fputws(L"ERROR: no Python found. The bundled runtime\\windows folder is\n"
               L"missing and no system Python is installed. Reinstall the package\n"
               L"or install Python from https://www.python.org/downloads/\n", stdout);
        pause_exit(1);
    }

    if (wcslen(python) + wcslen(L"serve.py") + 16 > PATHBUF + 64) {
        fputws(L"ERROR: package path is too long.\n", stdout);
        pause_exit(1);
    }
    cmd[0] = L'\0';
    wcscat(cmd, L"\"");
    wcscat(cmd, python);
    wcscat(cmd, L"\" serve.py");

    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        wprintf(L"ERROR: could not start Python (Windows error %lu):\n  %ls\n",
                GetLastError(), python);
        pause_exit(1);
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);

    if (code == 0)
        fputws(L"\nServer stopped.\n", stdout);
    else
        wprintf(L"\nThe server stopped with an error (exit code %lu).\n"
                L"See the messages above for details.\n", code);
    pause_exit((int)code);
    return (int)code; /* not reached */
}
