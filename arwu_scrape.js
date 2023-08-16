import puppeteer from 'puppeteer';
import fs from 'fs'

const allLinks = [];
const allNames = [];

const delay = ms => new Promise(res => setTimeout(res, ms));
// List of String containing the name-of-the-university::number-of-students (name-of-the-university::ERROR in case of error)
let scores = [];

const browser = await puppeteer.launch({headless:true});
const page = await browser.newPage();
  
// Navigate the page to a URL
await page.goto('https://www.shanghairanking.com/rankings/arwu/2022');

let i = 0;
while(true) {
    // Look for the div containing the number of students
    const searchResultSelector = 'tbody';
    // WAIT THE SELECTOR
    await page.waitForSelector(searchResultSelector);
    const element = await page.$(searchResultSelector);
    const arr = await element.$$eval('tr',(nodes)=>nodes.map(n=>
        {
            return n.children.item(1).children.item(0).children.item(1).children.item(0).children.item(0).children.item(0).innerHTML;
        })
    );

    const links = await element.$$eval('tr',(nodes)=>nodes.map(n=>
        {
            return n.children.item(1).children.item(0).children.item(1).children.item(0).children.item(0)["href"];
        })
    );

    const trimArray = arr.map(element => {
        return element.trim();
    });

    allLinks.push(...links);
    allNames.push(...trimArray)

    ++i;
    if (i == 35) {
        break;
    }
    // GO TO NEXT DATA
    const selectors = await page.$$('.ant-pagination-item-' + i)
    await selectors[0].click()
}


await browser.close();

const nameAndScore = [];

for(let i = 0; i < allLinks.length; ++i) {
    const link = allLinks[i];
    const browser = await puppeteer.launch({headless:true});
    try {
        const page = await browser.newPage();
    
        // Navigate the page to a URL
        await page.goto(link);

        const searchResultSelector = 'div > .num-box > span';
        await page.waitForSelector(searchResultSelector);
        const element = await page.$(searchResultSelector)
        const value = await page.evaluate(el => el.textContent, element);

        console.log(allNames[i] + "::" + value);
        nameAndScore.push(allNames[i] + "::" + value);
        await browser.close();
    } catch(error) {
        console.log("ERROR::", error);
        nameAndScore.push(allNames[i] + "::ERROR");
        await browser.close();
    }
}

fs.writeFile('ARWU-students.txt', nameAndScore.join('\n'), err => {
    if (err) {
      console.error(err);
    }
});