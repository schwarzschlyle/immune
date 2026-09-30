from __future__ import annotations

import re
import shlex
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

_SEPARATORS = frozenset({";", "&&", "||", "&", "\n"})
_WRAPPER_OPTIONS: dict[str, frozenset[str]] = {
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-U", "-r", "-t", "--user", "--group"}),
    "doas": frozenset({"-u", "-C"}),
    "env": frozenset({"-u", "-C", "-S", "--unset", "--chdir"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
    "nohup": frozenset(),
    "time": frozenset(),
    "exec": frozenset({"-a"}),
    "command": frozenset(),
}
_LEADING_POSITIONALS = {"timeout": 1}
_PIPE = "|"
_DOWNLOADERS = frozenset({"curl", "wget", "fetch", "iwr", "invoke-webrequest", "irm", "invoke-restmethod"})
_INTERPRETERS = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "ksh",
        "fish",
        "python",
        "python3",
        "node",
        "perl",
        "ruby",
        "php",
        "iex",
        "invoke-expression",
        "powershell",
        "pwsh",
    }
)
_READERS = frozenset(
    {
        "cat",
        "less",
        "more",
        "head",
        "tail",
        "cp",
        "scp",
        "rsync",
        "base64",
        "xxd",
        "od",
        "strings",
        "tar",
        "zip",
        "gpg",
        "openssl",
        "type",
        "get-content",
        "gc",
        "curl",
        "wget",
        "nc",
        "tee",
    }
)
_CREDENTIALS = re.compile(
    r"(?i)(?:^|[/\\~\s'\"=@])(?:\.ssh(?:[/\\]|$)|id_(?:rsa|ed25519|ecdsa|dsa)\b|\.aws[/\\]credentials|\.aws(?:[/\\]|$)|"
    r"\.env(?!\.(?:example|sample|template|dist)$)(?:\.[\w-]+)?$|\.netrc$|\.git-credentials$|\.docker[/\\]config\.json|\.kube[/\\]config|\.npmrc$|"
    r"\.pypirc$|/etc/(?:shadow|passwd|sudoers)\b|\.gnupg\b|login\.keychain|credentials\.json$)"
)
_ROOTS = frozenset(
    {
        "/",
        "/*",
        "~",
        "~/",
        "~/*",
        "$home",
        "${home}",
        "*",
        ".",
        "./",
        "./*",
        "..",
        "c:\\",
        "c:/",
        "/etc",
        "/usr",
        "/var",
        "/home",
        "/bin",
        "/boot",
        "/lib",
        "/opt",
        "/root",
        "/srv",
        "/system",
    }
)
_REVERSE = re.compile(
    r"(?i)/dev/(?:tcp|udp)/|\bnc(?:at)?\b[^\n|;]*\s-[ec]\s|\bsocat\b[^\n]*\bexec:|pty\.spawn|"
    r"socket\.socket\([^)]*\)[^\n]*(?:subprocess|os\.dup2)|net\.sockets\.tcpclient|\bmkfifo\b[^\n]*\bnc\b"
)
_POWERSHELL = re.compile(
    r"(?i)\b(?:iex|invoke-expression)\b[^\n]*(?:downloadstring|invoke-webrequest|iwr|irm|net\.webclient)|"
    r"(?:downloadstring|downloadfile)\s*\(|\s-(?:e|enc|encodedcommand)\s+[A-Za-z0-9+/=]{16,}|"
    r"set-executionpolicy\s+(?:unrestricted|bypass)|set-mppreference\s+-disable"
)
_DISABLE = re.compile(
    r"(?i)\bsetenforce\s+0\b|\bufw\s+disable\b|\biptables\s+-F\b|\bhistory\s+-c\b|\bunset\s+histfile\b|"
    r"\bsystemctl\s+(?:stop|disable)\s+(?:auditd|firewalld|apparmor)\b"
)
_WIPE = re.compile(
    r"(?i)\bmkfs(?:\.\w+)?\b|\bdd\b[^\n]*\bof=/dev/(?:sd|nvme|hd|disk)|:\(\)\s*\{\s*:\|:&\s*\};:|\bshred\b"
)
_URL = re.compile(r"(?i)^(?:git\+|https?://|github:|gitlab:|bitbucket:|git@|[\w.-]+/[\w.-]+#)")
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "iex", "invoke-expression", "powershell", "pwsh"})
_FILESYSTEM_ROOTS = frozenset({"/", "~", "~/", "$home", "${home}", "c:\\", "c:/"})
_PERSISTENCE = re.compile(r"(?i)(?:\bcrontab\b|\.bashrc\b|\.zshrc\b|\.profile\b|/etc/cron|launchagents)")
_FETCH = re.compile(r"(?i)\b(?:curl|wget|iwr|irm|invoke-webrequest)\b")


@dataclass(frozen=True, slots=True)
class Command:
    argv: tuple[str, ...]

    @property
    def program(self) -> str:
        for word in self.argv:
            if "=" in word and not word.startswith("-") and word.split("=", 1)[0].isidentifier():
                continue
            name = word.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
            return name.removesuffix(".exe")
        return ""

    @property
    def arguments(self) -> tuple[str, ...]:
        seen = False
        rest: list[str] = []
        for word in self.argv:
            if seen:
                rest.append(word)
            elif word.rsplit("/", 1)[-1].lower().removesuffix(".exe") == self.program:
                seen = True
        return tuple(rest)

    @property
    def unprivileged(self) -> Command:
        options = _WRAPPER_OPTIONS.get(self.program)
        if options is None:
            return self
        arguments = list(self.arguments)
        while arguments and (arguments[0].startswith("-") or "=" in arguments[0]):
            option = arguments.pop(0)
            if option in options and arguments:
                arguments.pop(0)
        del arguments[: _LEADING_POSITIONALS.get(self.program, 0)]
        return Command(tuple(arguments)).unprivileged if arguments else self


class ShellParser:
    def pipelines(self, text: str) -> list[list[Command]]:
        try:
            lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = ""
            words = list(lexer)
        except ValueError:
            words = text.split()
        pipelines: list[list[Command]] = [[]]
        current: list[str] = []
        for word in words:
            if word in _SEPARATORS:
                self._close(pipelines, current)
                pipelines.append([])
                current = []
            elif word in (_PIPE, "|&"):
                self._close(pipelines, current)
                current = []
            else:
                current.append(word)
        self._close(pipelines, current)
        return [pipeline for pipeline in pipelines if pipeline]

    @staticmethod
    def _close(pipelines: list[list[Command]], current: list[str]) -> None:
        if current:
            pipelines[-1].append(Command(tuple(current)).unprivileged)

    @staticmethod
    def nested(text: str, pipeline: Sequence[Command]) -> Iterator[str]:
        for match in re.finditer(
            r"\$\((?P<inner>[^()]{1,2000})\)|`(?P<tick>[^`]{1,2000})`|<\((?P<proc>[^()]{1,2000})\)", text
        ):
            yield match.group("inner") or match.group("tick") or match.group("proc") or ""
        for command in pipeline:
            arguments = command.arguments
            if command.program in _INTERPRETERS and "-c" in arguments:
                index = arguments.index("-c")
                if index + 1 < len(arguments):
                    yield arguments[index + 1]
            if command.program == "eval":
                yield " ".join(arguments)


class ShellAnalyzer:
    def __init__(self, parser: ShellParser | None = None, depth: int = 3) -> None:
        self._parser = parser or ShellParser()
        self._depth = depth

    def findings(self, text: str) -> set[str]:
        return self._analyze(text, self._depth)

    def _analyze(self, text: str, depth: int) -> set[str]:
        found: set[str] = set()
        for pattern, name in (
            (_REVERSE, "reverse_shell"),
            (_POWERSHELL, "powershell_download_exec"),
            (_DISABLE, "disable_security"),
            (_WIPE, "disk_wipe"),
        ):
            if pattern.search(text):
                found.add(name)
        substituted = re.search(r"(?:\$\(|`|<\()\s*(?:curl|wget|iwr|irm)\b", text, re.I)
        if _PERSISTENCE.search(text) and _FETCH.search(text):
            found.add("persistence")
        if re.search(r"\bremote\s+add\b", text) and re.search(r"\bgit\s+push\b", text):
            found.add("push_to_url")
        for pipeline in self._parser.pipelines(text):
            found.update(self._pipeline(pipeline))
            if substituted and any(command.program in _INTERPRETERS for command in pipeline):
                found.add("pipe_to_shell")
            if depth > 0:
                for inner in self._parser.nested(text, pipeline):
                    found.update(self._analyze(inner, depth - 1))
        return found

    def _pipeline(self, pipeline: Sequence[Command]) -> set[str]:
        found: set[str] = set()
        programs = [command.program for command in pipeline]
        downloads = [index for index, program in enumerate(programs) if program in _DOWNLOADERS]
        if downloads and any(self._runs_stdin(command) for command in pipeline[downloads[0] + 1 :]):
            found.add("pipe_to_shell")
        if "base64" in programs:
            decode = programs.index("base64")
            decoding = any(flag in ("-d", "--decode", "-D") for flag in pipeline[decode].arguments)
            if decoding and any(self._runs_stdin(command) for command in pipeline[decode + 1 :]):
                found.add("encoded_exec")
        for command in pipeline:
            found.update(self._command(command))
        if any(program in ("env", "printenv", "set") for program in programs) and downloads:
            found.add("environment_exfiltration")
        return found

    def _command(self, command: Command) -> set[str]:
        program, arguments = command.program, command.arguments
        found: set[str] = set()
        if program == "rm" and self._recursive(arguments) and any(self._root(target) for target in arguments):
            found.add("destructive_delete")
        if program in ("remove-item", "rmdir", "rd", "del") and any(self._root(target) for target in arguments):
            found.add("destructive_delete")
        if program == "find" and arguments[:1] and arguments[0].lower() in _FILESYSTEM_ROOTS and "-delete" in arguments:
            found.add("destructive_delete")
        if program in _READERS and any(_CREDENTIALS.search(target) for target in arguments):
            found.add("credential_read")
        if self._installs_from_url(program, arguments):
            found.add("install_from_url")
        if program == "cargo" and "install" in arguments and "--git" in arguments:
            found.add("install_from_url")
        if (
            program == "git"
            and arguments[:1] == ("push",)
            and any("://" in target or target.startswith("git@") for target in arguments)
        ):
            found.add("push_to_url")
        if program == "git" and arguments[:2] == ("remote", "set-url"):
            found.add("push_to_url")
        if program == "chmod" and any(mode in ("777", "a+rwx", "+s", "u+s", "g+s") for mode in arguments):
            found.add("unsafe_permissions")
        return found

    @staticmethod
    def _installs_from_url(program: str, arguments: Sequence[str]) -> bool:
        targets = [argument for argument in arguments if not argument.startswith("-")]
        if program in ("pip", "pip3", "uv", "pipx") and "install" in targets:
            return any(_URL.match(target) for target in targets)
        if program in ("npm", "pnpm", "yarn", "bun") and targets[:1] and targets[0] in ("install", "i", "add"):
            return any(_URL.match(target) for target in targets[1:])
        return False

    @staticmethod
    def _runs_stdin(command: Command) -> bool:
        program, arguments = command.program, command.arguments
        if program not in _INTERPRETERS:
            return False
        if any(flag in ("-m", "-c", "-e", "--version", "-V") for flag in arguments):
            return False
        positional = [argument for argument in arguments if not argument.startswith("-")]
        if program in _SHELLS:
            return not positional or positional[0] == "-" or "-s" in arguments
        return not positional or positional[0] == "-"

    @staticmethod
    def _recursive(arguments: Sequence[str]) -> bool:
        return any(
            (word.startswith("-") and not word.startswith("--") and "r" in word.lower()) or word == "--recursive"
            for word in arguments
        )

    @staticmethod
    def _root(target: str) -> bool:
        return target.lower().rstrip() in _ROOTS or target.lower().startswith(("/*", "~/*"))
