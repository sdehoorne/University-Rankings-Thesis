//CURL request to get data : curl --location --request GET 'https://www.topuniversities.com/rankings/endpoint?nid=3816281&page=0&items_per_page=1422&tab=indicators&region=&countries=&cities=&search=&star=&sort_by=overallscore&order_by=desc'



import { data } from './data.js'
import puppeteer from 'puppeteer';
import fs from 'fs'

// List of String containing the name-of-the-university::number-of-students (name-of-the-university::ERROR in case of error)
let scores = [];

const scrap = async (path, name) => {
    // Launch the browser and open a new blank page
    const browser = await puppeteer.launch({headless:true});
    try {
        const page = await browser.newPage();
  
        // Navigate the page to a URL
        await page.goto('https://www.topuniversities.com' + path);
    
        // Set screen size (USELESS HERE SINCE HEADLESS)
        await page.setViewport({width: 1080, height: 1024});
    
        // Look for the div containing the number of students
        const searchResultSelector = 'div > .studstaff-subsection-count';
        await page.waitForSelector(searchResultSelector);
        const element = await page.$(searchResultSelector)
        const value = await page.evaluate(el => el.textContent, element)
        console.log("DIV::", value);
        scores.push(name + "::" + value);

        await browser.close();
    } catch(error) {
        console.log("ERROR::", error);
        scores.push(name + "::ERROR");
        try {
            await browser.close();
        } catch(error_closing) {
            console.log("ERROR_CLOSING::", error_closing);
        }
    }
}

export const getData = async () => {
    console.log("GET DATA::")
    for (let i = 0; i < data.score_nodes.length; ++i) {
        const item = data.score_nodes[i];
        const path = item["path"]
        console.log("PATH::", i, "::", path)
        await scrap(path, item.title);
    }    
}


await getData();

//console.log("SCORE::", scores)
fs.writeFile('QS-students.txt', scores.join('\n'), err => {
  if (err) {
    console.error(err);
  }
  // fichier écrit avec succès
});
