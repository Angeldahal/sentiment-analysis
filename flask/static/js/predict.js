async function predict(input){
    const url = "/predict";
    let body;
    let headers = new Headers();

    body = JSON.stringify({
        text: input,
    });
    headers.append("Content-Type", "application/json");

    try {
        console.log(body, headers)
        const response = await fetch(url, {
            method: "POST",
            body: body,
            headers: headers,
        });

        if (!response.ok) throw new Error(response)
        const data = await response.json();

        document.getElementById("sentiment").innerText = data.sentiment;
        document.getElementById("probability").innerText = data.probability;
    } catch (error){
        console.log(error);
    }
}

$("#submit-btn").on("click", async function () {
    $("#submit-btn").prop("disabled", true);

    text = $("#user-input").val();
    console.log(text);
    await predict(text);

    $("#submit-btn").prop("disabled", false);
})