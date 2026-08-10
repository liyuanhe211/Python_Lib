// Py_Launcher.cpp
//
// A small Windows launcher executable.
//
// Behaviour:
//   1. Look at this .exe's own full path, e.g.
//          E:\...\src\Python_Lib\My_Lib_Plot_Editor.exe
//   2. Find the sibling .py file with the SAME directory + SAME base name:
//          E:\...\src\Python_Lib\My_Lib_Plot_Editor.py
//   3. Walk up the directory tree (starting from the .exe's own folder)
//      looking for a uv virtual environment (a ".venv" sub-directory).
//        - If found at <root>\.venv, run from <root>:
//              cd <root>
//              uv run --no-sync python "<pyfile>" <forwarded args...>
//          (--no-sync uses the existing .venv as-is, no dependency re-resolve)
//        - If no .venv is found all the way up to the drive root, fall back
//          to the system python, run from the .py file's own folder:
//              cd <pyfile_dir>
//              python "<pyfile>" <forwarded args...>
//   4. Any extra command-line arguments given to the .exe are forwarded to
//      the python script (so you can e.g. drag-and-drop a file onto the exe).
//
// The launcher waits for the child process and returns its exit code.
//
// Build (console subsystem, shows a console window / errors):
//     g++ -std=c++23 -O2 -static -o Py_Launcher.exe Py_Launcher.cpp
// Build (windowed subsystem, no console window — nicer for GUI tools):
//     g++ -std=c++23 -O2 -static -mwindows -DGUI_MODE -o Py_Launcher.exe Py_Launcher.cpp

#include <windows.h>
#include <shellapi.h>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Report a fatal error to the user and exit.
// In GUI mode this is a MessageBox; otherwise it is written to stderr.
[[noreturn]] static void fail(const std::wstring& msg) {
#ifdef GUI_MODE
    MessageBoxW(nullptr, msg.c_str(), L"Py_Launcher", MB_ICONERROR | MB_OK);
#else
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    std::wstring line = L"Py_Launcher error: " + msg + L"\n";
    DWORD written = 0;
    WriteConsoleW(h, line.c_str(), static_cast<DWORD>(line.size()), &written, nullptr);
#endif
    ExitProcess(1);
}

// Full path of this running executable.
static fs::path self_path() {
    std::vector<wchar_t> buf(32768);
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0) fail(L"GetModuleFileNameW failed");
    return fs::path(std::wstring(buf.data(), n));
}

// Quote a single argument according to the rules CommandLineToArgvW expects.
// (See "Everyone quotes command line arguments the wrong way", Daniel Colascione.)
static std::wstring quote_arg(const std::wstring& arg) {
    if (!arg.empty() &&
        arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;  // no quoting needed
    }
    std::wstring out = L"\"";
    for (auto it = arg.begin();; ++it) {
        unsigned backslashes = 0;
        while (it != arg.end() && *it == L'\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            out.append(backslashes * 2, L'\\');
            break;
        } else if (*it == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(*it);
        } else {
            out.append(backslashes, L'\\');
            out.push_back(*it);
        }
    }
    out.push_back(L'"');
    return out;
}

// Run the given command line in the given working directory; wait; return exit code.
static int run(const std::wstring& cmdline, const fs::path& cwd) {
    // CreateProcessW may modify the command-line buffer, so make it writable.
    std::wstring mutable_cmd = cmdline;

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    std::wstring cwd_str = cwd.wstring();

    BOOL ok = CreateProcessW(
        nullptr,                       // search PATH using the command line
        mutable_cmd.data(),
        nullptr, nullptr,
        FALSE,                         // do not inherit handles
        0,                             // share parent console (console build)
        nullptr,                       // inherit environment
        cwd_str.c_str(),               // working directory
        &si, &pi);

    if (!ok) {
        DWORD err = GetLastError();
        fail(L"Failed to launch:\n  " + cmdline +
             L"\nin directory:\n  " + cwd_str +
             L"\nWin32 error code: " + std::to_wstring(err) +
             L"\n(Is uv / python on your PATH?)");
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(code);
}

int main() {
    const fs::path exe = self_path();
    const fs::path exe_dir = exe.parent_path();
    const std::wstring stem = exe.stem().wstring();  // base name without extension

    // The python script we want to run: same folder, same base name, .py
    const fs::path py_file = exe_dir / (stem + L".py");
    if (!fs::exists(py_file)) {
        fail(L"No sibling python script found:\n  " + py_file.wstring());
    }

    // Walk up from the exe's directory looking for a ".venv" directory.
    fs::path venv_root;  // empty == not found
    std::error_code ec;
    for (fs::path dir = exe_dir;;) {
        if (fs::is_directory(dir / L".venv", ec)) {
            venv_root = dir;
            break;
        }
        const fs::path parent = dir.parent_path();
        if (parent == dir) break;  // reached the drive / filesystem root
        dir = parent;
    }

    // Collect forwarded command-line arguments (skip argv[0]).
    std::vector<std::wstring> fwd;
    {
        int argc = 0;
        LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (argv) {
            for (int i = 1; i < argc; ++i) fwd.emplace_back(argv[i]);
            LocalFree(argv);
        }
    }

    std::wstring cmd;
    fs::path cwd;
    if (!venv_root.empty()) {
        // uv run --no-sync python "<pyfile>" <args...>   (cwd = venv root)
        // --no-sync: use the existing .venv as-is and do NOT re-resolve /
        // re-install dependencies. This makes launching fast and robust even
        // if the project's pyproject.toml currently has unsatisfiable deps;
        // the goal here is "run this tool in its venv", not manage the env.
        cmd = L"uv run --no-sync python " + quote_arg(py_file.wstring());
        cwd = venv_root;
    } else {
        // python "<pyfile>" <args...>          (cwd = pyfile dir)
        cmd = L"python " + quote_arg(py_file.wstring());
        cwd = exe_dir;
    }
    for (const auto& a : fwd) {
        cmd += L' ';
        cmd += quote_arg(a);
    }

    return run(cmd, cwd);
}
