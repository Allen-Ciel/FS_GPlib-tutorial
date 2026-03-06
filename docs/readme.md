## 分支与发布

- **main**：发行版本。库代码、打 tag、`python -m build` 都在 main 上；Read the Docs 不从这里构建。
- **docs**：说明书开发分支。改 `docs/` 在这里做，推上去后由 Read the Docs 从 docs 分支构建。

**只更新说明书（日常）：**

```bash
git checkout docs
# 编辑 docs/ 下内容...
git add docs/
git commit -m "docs: 更新 xxx"
git push origin docs
```

**发新库版本（如 0.8.0）：**

```bash
git checkout main
# 可选：把 docs 分支最新说明合并进来
# git merge docs
git tag -a v0.8.0 -m "Release 0.8.0"
python -m build
git push origin main
git push origin v0.8.0
```

**让 docs 分支跟上 main 的发行代码（例如 API 变了）：**

```bash
git checkout docs
git merge main
# 若有冲突解决后...
git push origin docs
```

## upload

```bash
git push -u origin docs
```
上传到 docs 分支；Read the Docs 构建的也是 docs 分支。

## 本地

```ssh
sphinx-autobuild source build.html
```
可以在浏览器查看，并且自动更新修改内容。

```ssh
make html-all  # 手动构建英文 + 中文两个版本
make serve # 查看中英文两个版本
```

编写内容时用 `sphinx-autobuild` 获得实时预览的便利；需要测试双语切换效果时再跑 `make html-all` && `make serve`。


## 更新日志

- 8.26 修订了index.rst文件中的内容，Project Structure内容待定，这页之后视情况添加实验效果。
- 8.27 写了一点tutorial。
- 8.28 差不多写完了tutorial。
- 10.21 完成HK和SIR。SI和SIS基本完成（差参考文献）
- 10.22 基本完成SEIR，SEIRct，SEIS，SEISct
- 10.28 基本完成IC，LT（SWIR有问题）

- 3.2 添加双语切换模块（但并不能自动翻译，[国际化介绍](https://docs.readthedocs.com/platform/stable/localization.html)

