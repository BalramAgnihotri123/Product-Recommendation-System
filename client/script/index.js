const input = document.querySelector("#customerId");

input.addEventListener("keydown",async (e) => {
    if(e.key === "Enter")
        await getRecommendations()
})

const getRecommendations = async () => {
    if(!input.value) return;

    const ul = document.querySelector("#recommendations");
    const loaderEl = document.querySelector(".loader");
    const customerId = input.value;
    input.value = "";
    let date = "15/02/2023"
    let response;
    try{
        loaderEl.textContent = "Loading...";
        ul.innerHTML = "";
        response = await fetch(`http://localhost:5000/recommendations?customer_id=${customerId}&date=${date}`);
        if(!response.ok) {
            let responseText = await response.text()
            console.log(responseText)
            throw new Error(responseText);
        }
            
        const recommendations = await response.json();
        console.log({recommendations});

        recommendations.forEach((recommendation,index) => {
            let p = document.createElement("p");
            p.textContent = `-${recommendation}`;
            p.setAttribute("id",index)
            ul.appendChild(p)
        })
        loaderEl.textContent = "";
    } catch(err){
        loaderEl.textContent = err;
        console.log("failed to load recommendations ",JSON.stringify(err))
    }
}