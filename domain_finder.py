#!/usr/bin/env python3
"""
domain_finder.py — Company Domain Discovery Tool
=================================================
Aggregates domain intelligence from multiple passive and active sources:
  1. Certificate Transparency logs  (crt.sh)
  2. DNS record analysis            (A, MX, TXT/SPF, NS, CNAME)
  3. Reverse WHOIS pivot            (ViewDNS.info public API)
  4. Wayback Machine CDX API        (archived URLs → extra (sub)domains)
  5. SPF record chain walking       (follow include: directives)
  6. Built-in subdomain wordlist    (common names brute-force)
  7. External tool integration      (subfinder / amass / assetfinder if installed)
  8. Live HTTP probing              (validate which hosts respond)
  9. Deduplicated, sorted output    (console + optional file)

Usage:
  python3 domain_finder.py -d example.com [options]

Options:
  -d, --domain      Seed domain (required)
  -c, --company     Company / organisation name for WHOIS pivot (optional)
  -o, --output      Write results to this file (default: results.txt)
  --no-probe        Skip HTTP probing of discovered hosts
  --no-brute        Skip built-in subdomain brute-force
  --no-external     Skip external tools (subfinder/amass/assetfinder)
  --threads         Concurrent threads for HTTP probing  (default: 50)
  --timeout         HTTP probe timeout in seconds        (default: 5)
  -v, --verbose     Show debug / info messages

Requirements (pip install):
  requests
  dnspython

Optional system tools (not required but improve coverage):
  subfinder  https://github.com/projectdiscovery/subfinder
  amass      https://github.com/owasp-amass/amass
  assetfinder https://github.com/tomnomnom/assetfinder
"""

import argparse
import concurrent.futures
import json
import logging
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("[!] Missing dependency: pip install requests")

try:
    import dns.resolver
    import dns.reversename
except ImportError:
    sys.exit("[!] Missing dependency: pip install dnspython")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(levelname)s %(message)s",
    level=logging.WARNING,
)
log = logging.getLogger("domain_finder")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_THREADS = 50
DEFAULT_TIMEOUT = 5

COMMON_SUBDOMAINS = [
    "www", "mail", "smtp", "imap", "pop", "pop3", "webmail", "email",
    "mx", "mx1", "mx2", "ns", "ns1", "ns2", "dns", "dns1", "dns2",
    "ftp", "sftp", "ssh", "vpn", "remote", "citrix", "rdp",
    "dev", "stage", "staging", "test", "uat", "qa", "sandbox",
    "prod", "production", "api", "api2", "api-v2", "v2", "v3",
    "app", "apps", "web", "web2", "portal", "secure", "ssl",
    "admin", "panel", "dashboard", "control", "cp", "cpanel",
    "confluence", "jira", "gitlab", "github", "git", "bitbucket",
    "jenkins", "ci", "cd", "build", "deploy",
    "shop", "store", "ecom", "cart", "checkout", "pay", "payment",
    "cdn", "static", "assets", "media", "img", "images", "video",
    "upload", "download", "files", "storage", "backup",
    "blog", "news", "press", "events", "help", "support", "docs",
    "kb", "forum", "community", "status", "monitor", "health",
    "internal", "intranet", "extranet", "corp", "office",
    "careers", "jobs", "hr", "finance", "legal",
    "login", "auth", "sso", "oauth", "id", "account", "accounts",
    "autodiscover", "autoconfig", "exchange", "owa",
    "mobile", "m", "wap", "app", "ios", "android",
    "chat", "meet", "video", "conference", "webinar",
    "analytics", "tracking", "metrics", "stats", "data",
    "aws", "azure", "gcp", "cloud", "k8s", "docker",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DomainFinder/1.0; "
        "+https://github.com/elementalsouls/AI_Training)"
    )
}

# ---------------------------------------------------------------------------
# HTTP session factory (with retries)
# ---------------------------------------------------------------------------

def make_session(timeout: int = DEFAULT_TIMEOUT) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


# ---------------------------------------------------------------------------
# 1. Certificate Transparency — crt.sh
# ---------------------------------------------------------------------------

def query_crtsh(domain: str, session: requests.Session) -> set:
    """Return all (sub)domains found in crt.sh CT logs for the seed domain."""
    log.info("[crt.sh] Querying certificate transparency logs …")
    found = set()
    url = f"https://crt.sh/?q=%.{domain}&output=json"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            for entry in resp.json():
                for name in entry.get("name_value", "").splitlines():
                    name = name.strip().lstrip("*.")
                    if name and domain in name:
                        found.add(name.lower())
    except Exception as exc:
        log.warning("[crt.sh] Error: %s", exc)
    log.info("[crt.sh] Found %d entries", len(found))
    return found


# ---------------------------------------------------------------------------
# 2. DNS Record Analysis
# ---------------------------------------------------------------------------

def query_dns(domain: str) -> dict:
    """
    Resolve A, MX, TXT (SPF), NS, CNAME records for a domain.
    Returns a dict keyed by record type with lists of string values.
    """
    record_types = ["A", "MX", "TXT", "NS", "CNAME", "AAAA"]
    results = defaultdict(list)
    resolver = dns.resolver.Resolver()
    resolver.lifetime = 8
    for rtype in record_types:
        try:
            answers = resolver.resolve(domain, rtype)
            for rdata in answers:
                results[rtype].append(str(rdata).rstrip("."))
        except Exception:
            pass
    return dict(results)


def extract_spf_domains(spf_record: str) -> set:
    """Parse an SPF TXT record and extract all referenced domains."""
    found = set()
    # include:domain.com  redirect=domain.com  a:domain.com  mx:domain.com
    patterns = [
        r"include:([a-zA-Z0-9._-]+)",
        r"redirect=([a-zA-Z0-9._-]+)",
        r"\ba:([a-zA-Z0-9._-]+)",
        r"\bmx:([a-zA-Z0-9._-]+)",
        r"ip4:[0-9./]+",          # skip IPs
        r"ip6:[0-9a-fA-F:./]+",   # skip IPv6
    ]
    for pat in patterns[:4]:
        for match in re.findall(pat, spf_record, re.IGNORECASE):
            found.add(match.lower().rstrip("."))
    return found


def walk_spf(domain: str, visited: set | None = None, depth: int = 0) -> set:
    """Recursively follow SPF include: chains and collect all referenced domains."""
    if visited is None:
        visited = set()
    if domain in visited or depth > 5:
        return set()
    visited.add(domain)
    log.info("[SPF] Walking %s (depth=%d)", domain, depth)
    found = set()
    try:
        records = query_dns(domain).get("TXT", [])
        for txt in records:
            if "v=spf1" in txt.lower():
                refs = extract_spf_domains(txt)
                found.update(refs)
                for ref in refs:
                    found.update(walk_spf(ref, visited, depth + 1))
    except Exception as exc:
        log.warning("[SPF] Error walking %s: %s", domain, exc)
    return found


def dns_pivot(domain: str) -> dict:
    """
    Full DNS analysis: records + derived domain hints from MX / NS / SPF.
    Returns a dict with 'records' and 'related_domains'.
    """
    log.info("[DNS] Analysing %s", domain)
    records = query_dns(domain)
    related = set()

    # MX → mail servers may be on sibling or parent domains
    for mx in records.get("MX", []):
        mx_host = re.sub(r"^\d+\s+", "", mx).rstrip(".")
        related.add(mx_host.lower())
        parts = mx_host.split(".")
        if len(parts) >= 2:
            related.add(".".join(parts[-2:]).lower())

    # NS → nameserver parent domains
    for ns in records.get("NS", []):
        ns_host = ns.rstrip(".")
        related.add(ns_host.lower())

    # SPF chain
    spf_domains = walk_spf(domain)
    related.update(spf_domains)

    return {"records": records, "related_domains": related}


# ---------------------------------------------------------------------------
# 3. Reverse WHOIS via ViewDNS.info (free, no API key needed)
# ---------------------------------------------------------------------------

def reverse_whois(company_name: str, session: requests.Session) -> set:
    """
    Query ViewDNS.info reverse WHOIS for domains registered by the same
    organisation / email.  Returns a set of domain strings.
    """
    if not company_name:
        return set()
    log.info("[WHOIS] Reverse WHOIS for '%s' …", company_name)
    found = set()
    url = (
        f"https://viewdns.info/reversewhois/"
        f"?q={requests.utils.quote(company_name)}&output=json"
    )
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            domains = (
                data.get("response", {})
                    .get("domains", [])
            )
            for entry in domains:
                d = entry.get("domain", "").strip().lower()
                if d:
                    found.add(d)
    except Exception as exc:
        log.warning("[WHOIS] Error: %s", exc)
    log.info("[WHOIS] Found %d domains", len(found))
    return found


# ---------------------------------------------------------------------------
# 4. Wayback Machine CDX API
# ---------------------------------------------------------------------------

def wayback_domains(domain: str, session: requests.Session) -> set:
    """
    Query the Wayback CDX API for all URLs ever crawled under *domain*.
    Extract unique (sub)domain names from those URLs.
    """
    log.info("[Wayback] Querying archive.org CDX for %s …", domain)
    found = set()
    url = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url=*.{domain}/*&output=json&fl=original&collapse=urlkey&limit=5000"
    )
    try:
        resp = session.get(url, timeout=60)
        if resp.status_code == 200:
            for row in resp.json():
                if not row:
                    continue
                raw = row[0] if isinstance(row, list) else row
                try:
                    host = urlparse(raw).hostname or ""
                    if host and domain in host:
                        found.add(host.lower().lstrip("*."))
                except Exception:
                    pass
    except Exception as exc:
        log.warning("[Wayback] Error: %s", exc)
    log.info("[Wayback] Found %d (sub)domains", len(found))
    return found


# ---------------------------------------------------------------------------
# 5. External tool integration (subfinder / amass / assetfinder)
# ---------------------------------------------------------------------------

def run_external_tool(tool: str, domain: str) -> set:
    """Run an external subdomain enumeration tool and return results."""
    found = set()
    commands = {
        "subfinder": ["subfinder", "-d", domain, "-silent", "-all"],
        "amass":     ["amass", "enum", "-passive", "-d", domain],
        "assetfinder": ["assetfinder", "--subs-only", domain],
    }
    cmd = commands.get(tool)
    if not cmd:
        return found
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in proc.stdout.splitlines():
            line = line.strip().lower()
            if line and domain in line:
                found.add(line)
        log.info("[%s] Found %d subdomains", tool, len(found))
    except FileNotFoundError:
        log.debug("[%s] Not installed, skipping", tool)
    except subprocess.TimeoutExpired:
        log.warning("[%s] Timed out", tool)
    except Exception as exc:
        log.warning("[%s] Error: %s", tool, exc)
    return found


def run_all_external_tools(domain: str) -> set:
    found = set()
    for tool in ("subfinder", "amass", "assetfinder"):
        found.update(run_external_tool(tool, domain))
    return found


# ---------------------------------------------------------------------------
# 6. Built-in subdomain brute-force
# ---------------------------------------------------------------------------

def brute_force_subdomains(domain: str, wordlist: list | None = None) -> set:
    """
    Resolve common subdomain names against the seed domain.
    Returns only those that actually resolve.
    """
    if wordlist is None:
        wordlist = COMMON_SUBDOMAINS
    log.info("[Brute] Testing %d common subdomain names …", len(wordlist))
    found = set()
    resolver = dns.resolver.Resolver()
    resolver.lifetime = 3

    def resolve_sub(sub):
        fqdn = f"{sub}.{domain}"
        try:
            resolver.resolve(fqdn, "A")
            return fqdn
        except Exception:
            pass
        try:
            resolver.resolve(fqdn, "CNAME")
            return fqdn
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(resolve_sub, sub): sub for sub in wordlist}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                found.add(result.lower())

    log.info("[Brute] %d subdomains resolved", len(found))
    return found


# ---------------------------------------------------------------------------
# 7. Live HTTP probing
# ---------------------------------------------------------------------------

def probe_host(host: str, timeout: int) -> dict | None:
    """
    Probe a hostname over HTTP and HTTPS.
    Returns a dict with status_code, redirect, server header — or None.
    """
    for scheme in ("https", "http"):
        url = f"{scheme}://{host}"
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers=HEADERS,
                verify=False,
            )
            return {
                "host": host,
                "url": url,
                "status": resp.status_code,
                "redirect": resp.url if resp.url != url else None,
                "server": resp.headers.get("Server", ""),
                "title": _extract_title(resp.text),
            }
        except Exception:
            continue
    return None


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>([^<]{1,200})</title>", html, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def probe_all(hosts: set, threads: int = DEFAULT_THREADS, timeout: int = DEFAULT_TIMEOUT) -> list:
    """Probe a set of hostnames concurrently. Returns list of live-host dicts."""
    log.info("[Probe] Probing %d hosts …", len(hosts))
    live = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(probe_host, h, timeout): h for h in hosts}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                live.append(result)
    log.info("[Probe] %d live hosts", len(live))
    return sorted(live, key=lambda x: x["host"])


# ---------------------------------------------------------------------------
# 8. Output helpers
# ---------------------------------------------------------------------------

def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          Company Domain Discovery Tool v1.0              ║
║    Certificate Transparency · DNS · WHOIS · Wayback      ║
║    Brute-force · External tools · HTTP probing           ║
╚══════════════════════════════════════════════════════════╝
""")


def print_section(title: str, items):
    print(f"\n{'='*60}")
    print(f"  {title} ({len(items)} items)")
    print(f"{'='*60}")
    for item in sorted(items):
        print(f"  {item}")


def print_live(live: list):
    print(f"\n{'='*60}")
    print(f"  LIVE HOSTS ({len(live)} responding)")
    print(f"{'='*60}")
    for h in live:
        status = h.get("status", "?")
        server = h.get("server", "")
        title = h.get("title", "")
        redirect = h.get("redirect", "")
        line = f"  [{status}] {h['url']}"
        if server:
            line += f"  |  Server: {server}"
        if title:
            line += f"  |  Title: {title[:60]}"
        if redirect:
            line += f"  |  → {redirect}"
        print(line)


def save_results(
    domain: str,
    all_domains: set,
    all_subdomains: set,
    dns_info: dict,
    live: list,
    output_file: str,
):
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(f"# Domain Discovery Results for: {domain}\n")
        fh.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")

        fh.write("## Related Domains\n")
        for d in sorted(all_domains):
            fh.write(f"{d}\n")

        fh.write("\n## Subdomains\n")
        for s in sorted(all_subdomains):
            fh.write(f"{s}\n")

        fh.write("\n## DNS Records\n")
        records = dns_info.get("records", {})
        for rtype, values in sorted(records.items()):
            for v in values:
                fh.write(f"{rtype:10s} {v}\n")

        fh.write("\n## Live Hosts\n")
        for h in live:
            status = h.get("status", "?")
            title = h.get("title", "")
            fh.write(f"[{status}] {h['url']}")
            if title:
                fh.write(f"  # {title[:80]}")
            fh.write("\n")

    print(f"\n[+] Results written to: {output_file}")


# ---------------------------------------------------------------------------
# 9. Main orchestrator
# ---------------------------------------------------------------------------

def discover(
    domain: str,
    company: str = "",
    output_file: str = "results.txt",
    probe: bool = True,
    brute: bool = True,
    external: bool = True,
    threads: int = DEFAULT_THREADS,
    timeout: int = DEFAULT_TIMEOUT,
):
    banner()
    print(f"[*] Seed domain  : {domain}")
    if company:
        print(f"[*] Company name : {company}")
    print(f"[*] Output file  : {output_file}")
    print()

    # Suppress insecure HTTPS warnings from requests/urllib3
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = make_session(timeout)

    # ---- collect subdomains ------------------------------------------------
    all_subdomains: set = set()
    all_related: set = set()

    # 1. Certificate Transparency
    ct_results = query_crtsh(domain, session)
    all_subdomains.update(ct_results)
    print(f"[+] crt.sh            : {len(ct_results)} entries")

    # 2. DNS analysis
    dns_info = dns_pivot(domain)
    dns_related = dns_info.get("related_domains", set())
    all_related.update(dns_related)
    print(f"[+] DNS / SPF pivot   : {len(dns_related)} related domains")
    records = dns_info.get("records", {})
    for rtype, vals in records.items():
        log.info("  %s: %s", rtype, vals)

    # 3. Reverse WHOIS
    whois_domains = reverse_whois(company or domain, session)
    all_related.update(whois_domains)
    print(f"[+] Reverse WHOIS     : {len(whois_domains)} domains")

    # 4. Wayback Machine
    wb_results = wayback_domains(domain, session)
    all_subdomains.update(wb_results)
    print(f"[+] Wayback Machine   : {len(wb_results)} (sub)domains")

    # 5. External tools
    if external:
        ext_results = run_all_external_tools(domain)
        all_subdomains.update(ext_results)
        print(f"[+] External tools    : {len(ext_results)} subdomains")

    # 6. Brute-force
    if brute:
        brute_results = brute_force_subdomains(domain)
        all_subdomains.update(brute_results)
        print(f"[+] Brute-force       : {len(brute_results)} resolved subdomains")

    # ---- normalise / deduplicate ------------------------------------------
    # keep only hostnames that are subdomains of seed domain
    seed_subdomains = {
        h.lower().lstrip("*.")
        for h in all_subdomains
        if h.endswith(f".{domain}") or h == domain
    }
    seed_subdomains.add(domain)

    # separate out "other related domains" (not subdomains of seed)
    other_related = {
        h.lower()
        for h in all_related | all_subdomains
        if not (h.endswith(f".{domain}") or h == domain)
        and "." in h
    }

    total_unique = len(seed_subdomains) + len(other_related)
    print(f"\n{'─'*60}")
    print(f"[*] Total unique (sub)domains : {len(seed_subdomains)}")
    print(f"[*] Total related domains     : {len(other_related)}")
    print(f"[*] Grand total               : {total_unique}")

    # ---- print results ----------------------------------------------------
    print_section(f"SUBDOMAINS OF {domain}", seed_subdomains)
    if other_related:
        print_section("OTHER RELATED DOMAINS (DNS/WHOIS/SPF pivot)", other_related)

    # ---- live probing -----------------------------------------------------
    live = []
    if probe:
        print(f"\n[*] Probing {len(seed_subdomains)} hosts for live HTTP(S) …")
        live = probe_all(seed_subdomains, threads=threads, timeout=timeout)
        print_live(live)

    # ---- save results -----------------------------------------------------
    save_results(
        domain=domain,
        all_domains=other_related,
        all_subdomains=seed_subdomains,
        dns_info=dns_info,
        live=live,
        output_file=output_file,
    )

    return {
        "subdomains": seed_subdomains,
        "related_domains": other_related,
        "dns": dns_info,
        "live": live,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Company Domain Discovery Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-d", "--domain",   required=True,  help="Seed domain (e.g. example.com)")
    parser.add_argument("-c", "--company",  default="",     help="Company / org name for WHOIS pivot")
    parser.add_argument("-o", "--output",   default="results.txt", help="Output file (default: results.txt)")
    parser.add_argument("--no-probe",       action="store_true",   help="Skip HTTP probing")
    parser.add_argument("--no-brute",       action="store_true",   help="Skip subdomain brute-force")
    parser.add_argument("--no-external",    action="store_true",   help="Skip external tools")
    parser.add_argument("--threads",        type=int, default=DEFAULT_THREADS, help="HTTP probe threads")
    parser.add_argument("--timeout",        type=int, default=DEFAULT_TIMEOUT, help="HTTP probe timeout (s)")
    parser.add_argument("-v", "--verbose",  action="store_true",   help="Verbose / debug output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    discover(
        domain=args.domain,
        company=args.company,
        output_file=args.output,
        probe=not args.no_probe,
        brute=not args.no_brute,
        external=not args.no_external,
        threads=args.threads,
        timeout=args.timeout,
    )
