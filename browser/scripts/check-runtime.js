// Run in a browser with a usable WebGPU adapter to compare both execution paths.
async (page) => {
  await page.reload();
  await page.waitForFunction(() => !document.body.textContent.includes('Checking WebGPU'));
  const select = async (id, name) => { await page.locator(`#${id}`).press('ArrowDown'); await page.getByRole('option',{name,exact:true}).click(); };
  const run = async () => {
    await page.locator('#run-example').click();
    await page.waitForFunction(()=>!document.querySelector('#run').disabled,null,{timeout:120000});
    const result=JSON.parse(await page.locator('#response-json').textContent());
    if(result.error) throw Error(result.error);
    return result;
  };
  const gpu = page.getByRole('radio',{name:'WebGPU',exact:true});
  if(await gpu.isDisabled()) {
    return {webgpu:false, cpu:await run()};
  }
  await gpu.check();
  const rows=[];
  for(const task of ['Noul (Yes/No)','Choice','Score']) {
    await select('task',task);
    const result=await run();
    if(!(await page.locator('.result-time').textContent()).includes('WebGPU')) throw Error('Wrong backend label');
    rows.push({task,gpu:result});
  }
  await page.getByRole('radio',{name:'CPU',exact:true}).check();
  if(!(await page.locator('#model-status').textContent()).includes('Not downloaded')) throw Error('Switch must reset loaded state');
  for(const row of rows) {
    await select('task',row.task); row.cpu=await run();
    row.maxDifference=Math.max(...Object.keys(row.cpu.probabilities).map(id=>Math.abs(row.cpu.probabilities[id]-row.gpu.probabilities[id])));
    if(row.maxDifference > 0.001) throw Error(`CPU/WebGPU mismatch: ${row.maxDifference}`);
  }
  return {webgpu:true,rows};
}
