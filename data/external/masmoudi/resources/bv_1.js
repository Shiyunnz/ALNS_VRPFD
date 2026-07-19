(function() {
    const freemiumAd = document.getElementById("FreemiumAd");
    const freemiumAdDetail = document.getElementById("FreemiumAdDetailWrapper");
    const mobileAdClassName = "wsMobileAd";
    const desktopAdClassName = "wsDesktopAd";
    const useMediaQuery = true;

    freemiumAd.addEventListener("mouseenter", () => {
        if (!freemiumAd.classList.contains(mobileAdClassName)) {
            freemiumAd.style.maxHeight = "235px";
        }
    })
    freemiumAd.addEventListener("mouseleave", () => {
        if (!freemiumAd.classList.contains(mobileAdClassName)) {
            freemiumAd.style.maxHeight = "85px";
        }
    });

    if (useMediaQuery) {
        const mobileMediaQueryText = "screen and (max-width: 500px)";
        const mobileMediaQuery = matchMedia(mobileMediaQueryText);

        function handleMediaChange(mediaQuery) {
            if (mediaQuery.matches) {
                freemiumAd.classList.add(mobileAdClassName);
                freemiumAd.classList.remove(desktopAdClassName);
            } else {
                freemiumAd.classList.add(desktopAdClassName);
                freemiumAd.classList.remove(mobileAdClassName);
            }
        }

        mobileMediaQuery.addEventListener("change", handleMediaChange);

        // Initial check (if not done, the browser user agent will be considered to apply the appropriate CSS class)
        // handleMediaChange(mobileMediaQuery);
    }
})();