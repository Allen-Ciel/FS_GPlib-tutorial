
## upload

```ssh
git push -u origin docs
```
上传到docs分支，设置构建的也是docs分支

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

