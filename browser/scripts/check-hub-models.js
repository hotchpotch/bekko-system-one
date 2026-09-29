// Open the preview, then run with playwright-cli. Downloads both public models.
async (page) => {
  await page.reload();
  await page.getByRole("radio", {name:"CPU", exact:true}).click();
  const select=async(id,name)=>{await page.locator(`#${id}`).press('ArrowDown');await page.getByRole('option',{name,exact:true}).click();};
  const rows=[];
  for(const model of ['Bekko 17M · 29 MB','Bekko 68M · 196 MB']) {
    await select('model',model);
    if(!(await page.locator('#model-status').textContent()).includes('Not loaded')) throw Error('Model switch must clear session');
    for(const task of ['Noul (Yes/No)','Choice','Score']) {
      await select('task',task); await page.locator('#run-example').click();
      await page.waitForFunction(()=>!document.querySelector('#run').disabled,null,{timeout:180000});
      const result=JSON.parse(await page.locator('#response-json').textContent());
      if(result.error) throw Error(result.error);
      const values=Object.values(result.probabilities);
      if(!values.every(v=>Number.isFinite(v)&&v>=0&&v<=1)||Math.abs(values.reduce((a,b)=>a+b,0)-1)>1e-6) throw Error('Invalid probabilities');
      rows.push({model,task,result,status:await page.locator('#model-status').textContent()});
    }
  }
  return rows;
}
