"""Actual frontend workflows in isolated Chrome; real detectors, speech disabled.

Automation evidence only: no fabricated human participants or audible trials.
Start src/webapp.py --no-depth --no-tts --port 5057 first.
"""
import asyncio
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]


async def main():
    from playwright.async_api import async_playwright
    out=ROOT/'artifacts/four-class-study/browser';out.mkdir(exist_ok=True)
    invalid=out/'corrupt.jpg';invalid.write_bytes(b'not an image')
    badvideo=out/'corrupt.mp4';badvideo.write_bytes(b'not a video')
    fixture=ROOT/'demo/Stair/validation-only-R007.jpg'
    if fixture.exists():raise ValueError('Refusing to overwrite demo fixture')
    shutil.copy2(ROOT/'data/four_class_external/R007.jpg',fixture)
    report=dict(overall_passed=False,driver='Playwright isolated Chrome; not Windows UI helper',human_participants=0,physical_distance_tested=False,audibility_tested=False,flows=[],javascript_errors=[])
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=await browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda error:report['javascript_errors'].append(str(error)))
        async def status():return await (await page.request.get('http://127.0.0.1:5057/status')).json()
        async def mark(name,**details):report['flows'].append(dict(name=name,passed=True,**details))
        try:
            s=await status();assert not s['speech_enabled'],'Refuse to test with live speech enabled'
            await page.request.post('http://127.0.0.1:5057/control',data={'muted':True,'paused':True})
            await page.goto('http://127.0.0.1:5057/',wait_until='domcontentloaded')
            await page.wait_for_function("document.querySelectorAll('#legend li').length===5")
            assert await page.locator('#expected option').all_text_contents()==['Not set','Escalator','People','Stair','Wet Floor Sign']
            assert 'Pillar' not in await page.locator('footer').inner_text()
            assert await page.locator('#test-voice').is_disabled()
            await mark('four_class_controls_and_speech_disabled')
            await page.locator('h1').click();await page.keyboard.press('Space')
            await page.wait_for_function("document.querySelector('#pause').textContent.includes('Pause')")
            assert not (await status())['paused']
            await page.keyboard.press('Space');await page.wait_for_function("document.querySelector('#pause').textContent.includes('Resume')")
            assert (await status())['paused']
            await page.locator('#expected').focus();before=(await status())['muted'];await page.keyboard.press('m');assert (await status())['muted']==before
            await mark('keyboard_pause_resume_and_form_focus_guard')
            async with page.expect_response(lambda r:r.url.endswith('/analyze') and r.request.method=='POST') as pending:
                async with page.expect_file_chooser() as chooser:
                    await page.locator('#choose-images').click()
                await (await chooser.value).set_files([str(ROOT/'data/four_class_external/R007.jpg'),str(ROOT/'data/four_class_external/R002.jpg'),str(invalid)])
            response=await pending.value;assert response.status==200;batch=await response.json();assert len(batch['results'])==3 and 'error' in batch['results'][2]
            await page.wait_for_selector('#gallery .result-card');assert await page.locator('#gallery .result-card').count()==3
            await page.screenshot(path=str(out/'image-batch.png'),full_page=True)
            await mark('native_file_chooser_mixed_valid_corrupt_batch',results=3,real_predictions=True)
            await page.locator('#aview-alert').click();await page.wait_for_function("document.querySelector('#aview-alert').getAttribute('aria-pressed')==='true'")
            await page.locator('#aview-recognition').click();await mark('recognition_alert_view_switch')
            await page.locator('#back-live').click();await page.wait_for_function("document.querySelector('#view-badge').hidden")
            await page.locator('#tab-library').click();await page.locator('#library-reload').click()
            button=page.get_by_role('button',name='Analyse validation-only-R007.jpg (Stair)',exact=True);await button.wait_for()
            async with page.expect_response(lambda r:r.url.endswith('/analyze') and r.request.method=='POST') as pending:await button.click()
            assert (await pending.value).status==200
            await page.wait_for_function("!document.querySelector('#view-badge').hidden")
            await mark('demo_thumbnail_and_one_click_analysis')
            await page.locator('#back-live').click();await page.locator('#tab-video').click()
            async with page.expect_response(lambda r:r.url.endswith('/source') and r.request.method=='POST') as pending:
                await page.locator('#video-file').set_input_files(str(badvideo))
            invalid_response=await pending.value
            assert invalid_response.status in (400,415), (invalid_response.status,await invalid_response.text())
            assert (await invalid_response.json()).get('error')
            assert (await status())['source']['type']=='camera'
            await mark('corrupt_video_rejected_and_source_preserved')
            async with page.expect_response(lambda r:r.url.endswith('/source') and r.request.method=='POST',timeout=30000) as pending:
                async with page.expect_file_chooser() as chooser:await page.locator('#choose-video').click()
                await (await chooser.value).set_files(str(ROOT/'data/four_class_videos/opencv-vtest-first20s.mp4'))
            assert (await pending.value).status==200
            await page.wait_for_function("!document.querySelector('#source-bar').hidden")
            if (await status())['paused']:await page.locator('#pause').click()
            await page.locator('#vview-recognition').click()
            await page.wait_for_function("document.querySelector('#vview-recognition').getAttribute('aria-pressed')==='true'")
            await page.locator('#pause').click();await page.wait_for_function("document.querySelector('#pause').textContent.includes('Resume')")
            at=(await status())['source']['position'];await page.wait_for_timeout(650);assert abs((await status())['source']['position']-at)<=.2
            await page.locator('#pause').click()
            await page.wait_for_function("!document.querySelector('#video-summary').hidden",timeout=45000)
            final=await status();assert [x['class'] for x in final['video_summary']]==['Escalator','People','Stair','Wet Floor Sign']
            assert 'Pillar' not in await page.locator('#summary-note').inner_text()
            await page.screenshot(path=str(out/'video-summary.png'),full_page=True)
            await mark('real_video_upload_pause_resume_view_and_four_class_summary',frames=final['frames_processed'])
            await page.locator('#summary-replay').click();await page.wait_for_function("document.querySelector('#video-summary').hidden")
            await page.locator('#video-camera').click();await page.wait_for_function("document.querySelector('#source-bar').hidden")
            await page.request.post('http://127.0.0.1:5057/control',data={'paused':True,'muted':True})
            await mark('replay_and_return_to_camera')
            await page.locator('#tab-images').click();await page.set_viewport_size({'width':390,'height':844})
            await page.screenshot(path=str(out/'mobile-layout.png'),full_page=True)
            overflow=await page.evaluate('document.documentElement.scrollWidth-window.innerWidth')
            report['mobile_horizontal_overflow_px']=overflow
            assert overflow<=1, f'Mobile horizontal overflow: {overflow}px'
            await mark('mobile_layout_check',overflow_px=overflow)
            assert not report['javascript_errors'],report['javascript_errors']
            await mark('no_uncaught_frontend_errors')
            report['overall_passed']=True
        finally:
            await browser.close()
            assert fixture.resolve().is_relative_to((ROOT/'demo').resolve());fixture.unlink(missing_ok=True)
            report['temporary_demo_fixture_removed']=not fixture.exists()
            (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
