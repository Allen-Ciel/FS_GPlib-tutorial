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

    // Pattern: /build/<lang>/html/...  (local) or /<lang>/...  (hosted)
    var localRe  = /\/(build\/)(en|zh_CN)(\/html\/)/;
    var hostedRe = /\/(en|zh_CN)(\/)/;

    if (localRe.test(path)) {
        window.location.pathname = path.replace(localRe, '/$1' + targetLang + '$3');
    } else if (hostedRe.test(path)) {
        window.location.pathname = path.replace(hostedRe, '/' + targetLang + '$2');
    } else {
        // Fallback: try replacing one directory level up
        window.location.href = '../' + targetLang + '/html/index.html';
    }
}