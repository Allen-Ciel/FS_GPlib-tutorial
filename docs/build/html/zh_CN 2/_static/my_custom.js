$(document).ready(function(){
    let div_logo = document.getElementsByClassName("wy-side-nav-search")[0];
    let a_logo = div_logo.getElementsByTagName("a");
    a_logo[0].href = "https://github.com/Allen-Ciel/FS_GPlib-tutorial";
    a_logo[0].target = "_blank";
});

function switchLanguage(targetLang) {
    var defined = { 'en': true, 'zh_CN': true };
    if (!defined[targetLang]) return;

    var path = window.location.pathname;
    var langRe = /\/(en|zh_CN)\//;

    if (langRe.test(path)) {
        window.location.pathname = path.replace(langRe, '/' + targetLang + '/');
    }
}