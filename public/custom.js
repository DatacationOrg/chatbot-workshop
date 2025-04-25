function waitForElement(selector) {
    return new Promise(resolve => {
        if (document.querySelector(selector)) resolve(document.querySelector(selector));

        const observer = new MutationObserver(_ => {
            if (document.querySelector(selector)) {
                observer.disconnect();
                resolve(document.querySelector(selector));
            }
        });

        // If you get "parameter 1 is not of type 'Node'" error, see https://stackoverflow.com/a/77855838/492336
        observer.observe(document.body, { childList: true, subtree: true });
    });
}


replaceChainlitLogo = () => waitForElement("a[href='https://chainlit.io']").then((e) => e.closest("*").innerHTML = `
<a href="https://www.datacation.nl/" target="_blank" style="display: flex; align-items: center; text-decoration: none;">
    <p style="color: #616161; font-size: 12px; line-height: 1.5; margin: 0">
        <span>Developed by <b>Datacation B.V.</b></span>
    </p>
</a>`).then(() => setTimeout(replaceChainlitLogo, 1000));

document.addEventListener("DOMContentLoaded", replaceChainlitLogo);