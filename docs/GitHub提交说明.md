# GitHub 提交说明（本机实测配置）

本文档记录本机访问 GitHub 的**实测可用配置**。这些坑都已排查过，照抄即可，不必重新试错。

---

## 一、本机的网络状况（实测结论）

| 现象 | 实测结果 |
|---|---|
| 直接访问 github.com | ❌ 域名解析被污染（解析到境外无关 IP），TCP 443 连接超时 |
| 系统代理 | ✅ 已启用，地址 `127.0.0.1:7897`，浏览器可正常访问 GitHub |
| 命令行工具 | ❌ **默认不读取系统代理**，必须显式指定 |
| `curl.exe` | ❌ 报 `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS` |
| `git`（默认 schannel 后端） | ❌ 同样的凭证错误，无法访问远端 |
| `git`（改为 OpenSSL 后端） | ✅ **实测可用** |

**核心结论：GitHub 不是"上不去"，而是"命令行没走代理 + TLS 后端有毛病"。两处都修好即可。**

---

## 二、Git 的安装

本机原先没有 git，且系统自带的 `winget` 不可用。已部署**免安装版**（PortableGit，无需管理员权限）：

```
D:\Users\harness\_tools\PortableGit\cmd\git.exe
```

如果你希望在任何目录都能直接用 `git` 命令，把下面这个目录加入系统环境变量 `PATH`：

```
D:\Users\harness\_tools\PortableGit\cmd
```

---

## 三、必须做的两项 Git 配置

```bat
:: 1) TLS 后端改成 OpenSSL（关键！schannel 在本机拿不到凭证）
git config --global http.sslBackend openssl

:: 2) 让 git 走系统代理
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

:: 3) 你的身份信息（推送到 GitHub 时必需，请填入你自己的）
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
```

> 本项目的仓库里已经写好前两项（存在 `.git/config`，不会随仓库上传），
> 但**新建仓库或换机器时需要重新设置**。
>
> 代理端口 `7897` 是当前 Clash 类工具的监听端口，若你的工具改过端口，请同步修改。

### 验证是否配好

```bat
git ls-remote --heads https://github.com/gokriznastic/HybridSN
```

能打印出一行 `<哈希>  refs/heads/master` 就说明通了。

---

## 四、把本项目推送到你自己的 GitHub 仓库

### 1. 在 GitHub 网页上新建空仓库

仓库名建议 `HybridSN-HSI-Classification`，**不要**勾选 "Add a README file"（避免冲突）。

### 2. 本地关联并推送

```bat
cd /d D:\Users\harness\HybridSN-HSI-Classification

git remote add origin https://github.com/你的用户名/HybridSN-HSI-Classification.git
git branch -M main
git push -u origin main
```

首次推送会要求登录。GitHub 已不支持密码，请使用 **Personal Access Token**：

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 生成一个带 `repo` 权限的 token，复制保存
3. 推送时用户名填你的 GitHub 用户名，**密码处粘贴该 token**

### 3. 之后每次提交

```bat
git add -A
git commit -m "完成 5%/10%/30% 训练比例消融实验"
git push
```

---

## 五、注意事项

- **不要把数据集传上去**。`data/*.mat` 已在 `.gitignore` 中排除，README 里给出了数据来源与放置路径即可，这符合任务书"不建议上传大型原始数据集"的要求。
- **模型权重 `results/**/*.pth` 也已排除**，但 `results/` 下的曲线、混淆矩阵、对比表**要传**，它们是任务书要求的结果文件。
- 若哪天代理关了或换了端口，`git` 会因为连不上而报错，届时同步修改 `http.proxy` 即可。
- 提交前先 `git status` 看一眼，确认没有把 `.venv/`、`__pycache__/`、`.mplconfig/` 这类目录加进去。
