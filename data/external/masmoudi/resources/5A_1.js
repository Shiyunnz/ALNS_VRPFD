// Currently, the only major browser not supported is Internet Explorer

function UnsupportedBrowserCheck(ProjectName){

    function IsIE() {
        // document.documentMode for more recent IE and MSIE in the user agent for older IE
        return navigator.userAgent.indexOf("MSIE") !== -1 || !!document.documentMode;
    }

    function BrowserUpdateRedirect() {
        location.replace(location.protocol + "//" + location.host + "/Pages/UnsupportedBrowser");
    }

    function BrowserUpdatePopUp() {
        document.getElementById('alert-overlay').style.display = "block";
    }

    if (IsIE()) {
        if (ProjectName === 'Viewer') {
            BrowserUpdateRedirect();
        } else if (ProjectName === 'Editor' || ProjectName === 'BlogManagement' || ProjectName === 'Panel') {
            BrowserUpdatePopUp();
        }
    }
}

if (window.WS && WS.ProjectName) {
    UnsupportedBrowserCheck(WS.ProjectName);
}