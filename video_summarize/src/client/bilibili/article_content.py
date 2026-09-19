"""把 summary_bilibili.md 转成 B 站专栏草稿所需的正文 HTML。

B 站专栏正文是 HTML（不是 markdown）。B 站编辑器和微信一样：会丢弃 <style> 块，
只保留元素上的内联 style="..."。早期版本直接把 markdown 渲染成「裸 HTML」上传，
没有任何样式，于是标题没有橙色边框、行距错乱——和「在浏览器里全选公众号 HTML→
复制→粘贴进 B 站编辑器」得到的效果完全不同。

这里复刻手动复制粘贴的效果：
  1. 用与 summary_gongzhonghao.html 完全相同的 #wenyan 样式表渲染 markdown；
  2. premailer 把 <style> 规则下沉为元素内联 style（B 站只认内联样式）；
  3. 折叠标签间纯空白换行（浏览器渲染时本就折叠，否则 B 站会解析出空列表项/空行）；
  4. 去掉最外层 #wenyan 容器的 padding/margin/max-width（手动复制只取容器*内容*，
     外层 section 的页面级边距不会进剪贴板，留着会把全文向右挤出缩进）。

与公众号版唯一的不同：B 站版的「短期机会速览 / 个股观点速览 / 个股事件速览」等用的是本地 PNG 表格图
（![](N_xxx.png)），需要把这些本地图片收集出来，交上层上传 B 站图床后回填 src。

返回 (html, local_images)：
  - html：CSS 已内联、空白已折叠、外层布局边距已去除的正文 HTML，<img> 仍是本地相对路径
  - local_images：去重后的本地图片绝对路径列表（按出现顺序），等待上层上传替换
"""

import re
from pathlib import Path
from urllib.parse import unquote

import cssutils
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from premailer import transform

# premailer 内部用 cssutils 校验 CSS，而 cssutils 自带的属性表未收录 overflow-wrap /
# word-wrap（二者均为 CSS Text Module Level 3 的合法属性），于是把它们当未知属性刷 WARNING。
# 从根源解决：按 CSS 规范把这两个属性注册进 cssutils 的属性 profile，使其被识别为合法值。
cssutils.profile.addProfile(
    "CSS Text Module Level 3 (overflow-wrap)",
    {
        "overflow-wrap": "normal|break-word|anywhere|inherit",
        "word-wrap": "normal|break-word|anywhere|inherit",
    },
)

# 复用公众号版的样式表与外层模板，保证 B 站正文与「复制公众号 HTML」的观感完全一致
from src.rendering.rendering_gongzhonghao import _GZH_CSS, _GZH_HTML_TEMPLATE

# 仅出现在最外层正文容器上、用于页面级排版的属性；在编辑器内反而造成缩进 / 限宽
_WRAPPER_LAYOUT_PROPS = ("padding", "margin", "max-width")


def markdown_to_article_html(md_path: Path) -> tuple[str, list[Path]]:
    """读取 markdown 文件，渲染为带内联样式的专栏正文 HTML，并收集本地图片路径。

    Args:
        md_path: summary_bilibili.md 路径

    Returns:
        (html, local_images)
    """
    md_text = md_path.read_text(encoding="utf-8-sig")

    # html=True 让文中已有的 <h1 align="center"> 等内联 HTML 原样透传
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    content_html = md.render(md_text)

    # 收集本地图片（在内联/清洗前，src 仍是原始相对路径，最稳妥）
    local_images = _collect_local_images(content_html, md_path.parent)

    # 套上与公众号版完全相同的 #wenyan 样式外壳
    full_html = _GZH_HTML_TEMPLATE.replace("{css}", _GZH_CSS).replace("{content}", content_html)

    # 复刻手动复制粘贴效果（与 wechat/html_inline.py 同套处理）
    inlined = transform(full_html, keep_style_tags=False, remove_classes=False)
    inlined = re.sub(r">\s+<", "><", inlined)          # 折叠标签间空白
    inlined = _strip_wrapper_layout(inlined)            # 去最外层布局边距

    soup = BeautifulSoup(inlined, "lxml")
    _unwrap_image_paragraphs(soup)                      # 拆掉只含一张图的 <p>，去标题-图片间空行

    inner_html = _inner_html_of_wrapper(soup)
    return inner_html, local_images


def _collect_local_images(html: str, base_dir: Path) -> list[Path]:
    """从渲染后的 HTML 中收集本地图片的绝对路径（去重，保持出现顺序）。"""
    soup = BeautifulSoup(html, "lxml")
    local_images: list[Path] = []
    seen: set[str] = set()
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src or src.startswith(("http://", "https://", "//")):
            continue
        # markdown-it 会把中文文件名 URL 编码（%E7%9F...），需解码回真实路径
        img_path = (base_dir / unquote(src)).resolve()
        if str(img_path) not in seen:
            seen.add(str(img_path))
            local_images.append(img_path)
    return local_images


def _unwrap_image_paragraphs(soup: BeautifulSoup) -> None:
    """把「只包含一张图片」的 <p> 标签替换为其中的 <img>，去掉多余的段落间距。"""
    for p in soup.find_all("p"):
        children = [c for c in p.contents if not (isinstance(c, str) and not c.strip())]
        if len(children) == 1 and getattr(children[0], "name", None) == "img":
            p.replace_with(children[0])


def _strip_wrapper_layout(html: str) -> str:
    """删除最外层正文容器（#wenyan <section>）style 里的 padding/margin/max-width。

    premailer 在 style 值含双引号（如 font-family 的 "PingFang SC"）时会用单引号包裹属性，
    所以要同时兼容单/双引号两种 style 属性写法。
    """

    def rewrite_style(style_body: str) -> str:
        kept = [
            decl
            for decl in style_body.split(";")
            if decl.strip()
            and decl.split(":", 1)[0].strip().lower() not in _WRAPPER_LAYOUT_PROPS
        ]
        return "; ".join(kept)

    def repl(m: re.Match) -> str:
        quote = m.group(2)
        new_body = rewrite_style(m.group(3))
        style_attr = f" style={quote}{new_body}{quote}" if new_body else ""
        return f"<section{m.group(1)}{style_attr}{m.group(4)}>"

    return re.sub(
        r"<section((?:(?!\sstyle=)[^>])*)\sstyle=(['\"])(.*?)\2([^>]*)>",
        repl,
        html,
        count=1,
    )


def _inner_html_of_wrapper(soup: BeautifulSoup) -> str:
    """取最外层 #wenyan <section> 的内部 HTML（编辑器粘贴的是容器内容，不含容器本身）。

    找不到该容器时退回 body 内部 HTML。
    """
    wrapper = soup.find("section", id="wenyan")
    if wrapper is not None:
        return "".join(str(c) for c in wrapper.contents)
    body = soup.body
    return "".join(str(c) for c in body.contents) if body else str(soup)


def replace_image_src(html: str, url_map: dict[str, str]) -> str:
    """把正文 HTML 中本地图片的 src 替换为图床 URL。

    Args:
        html: markdown_to_article_html 产出的 HTML
        url_map: {本地图片绝对路径字符串: 图床URL}

    Returns:
        替换后的 HTML
    """
    soup = BeautifulSoup(html, "lxml")
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src or src.startswith(("http://", "https://", "//")):
            continue
        src_name = Path(unquote(src)).name
        matched = None
        for local_path, url in url_map.items():
            if Path(local_path).name == src_name:
                matched = url
                break
        if matched:
            img["src"] = matched

    body = soup.body
    inner_html = "".join(str(c) for c in body.contents) if body else str(soup)
    return inner_html
