#!/usr/bin/env python

"""
spica - chat ui for local llms

© 2023 hexis systems GmbH
Licensed under the MIT License.
"""

import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime
from llama_cpp import Llama, LlamaGrammar
from typing import AsyncGenerator, Callable, Iterable, Optional, Union, List, TypeVar, cast
from langstream.core.stream import Stream, StreamOutput
import flet as ft

import logging
logging.basicConfig(level=logging.DEBUG)


# globals

T = TypeVar('T')
U = TypeVar('U')
loop = None

class LLamaCppStream(Stream[T, U]):
    # defaults cf. https://github.com/abetlen/llama-cpp-python/blob/4177ae6d3482bb8af57b85cc7183ca83bd1571c4/examples/low_level_api/common.py#L12
    def __init__(
        self: 'LLamaCppStream[T, str]',
        name: str,
        call: Callable[
            [T],
            str,
        ],
        model: str,
        n_ctx: int = 4096,  # llama2: 4096
        n_batch: int = 8,
        n_threads: Optional[int] = None,
        max_tokens: int = -1,  # <0 = n_ctx
        temperature: float = 0.8,
        stop: Optional[Union[str, List[str]]] = [],
        top_k: float = 40,
        top_p: float = 0.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.18,
        grammar: LlamaGrammar = None,
        #running: bool = True
    ) -> None:

        llm = Llama(model_path=model, verbose=False, n_ctx=n_ctx, n_batch=n_batch, n_threads=n_threads)
        global llamagrammar
        llamagrammar = grammar

        async def generate(prompt: str) -> AsyncGenerator[U, None]:
            global loop
            loop = asyncio.get_event_loop()

            def get_outputs() -> Iterable[str]:
                global llamagrammar
                stream = llm.create_completion(
                    prompt,
                    stream=True,
                    echo=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    top_k=top_k,
                    top_p=top_p,
                    tfs_z=tfs_z,
                    mirostat_mode=mirostat_mode,
                    mirostat_tau=mirostat_tau,
                    mirostat_eta=mirostat_eta,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repeat_penalty=repeat_penalty,
                    grammar=llamagrammar
                )
                for output in stream:
                    if not self.running:
                        raise Exception('Generation canceled.')
                    yield output['choices'][0]['text']

            outputs = await loop.run_in_executor(None, get_outputs)

            for output in outputs:
                yield cast(U, output)
        
        super().__init__(name, lambda input: generate(call(input)))
    
    running = True

class Message():
    def __init__(self, user: str, text: str):
        self.user = user
        self.text = text

async def stream_output(page, chat_stream, message, mode='chat', **kwargs):
    condense_output = ''
    
    async for output in chat_stream(message):
        if mode == 'chat':
            page.pubsub.send_all_on_topic('stream', output)
        elif mode == 'condense':
            condense_output += output.data
        else:
            await output

    if mode == 'chat':
        page.pubsub.send_all_on_topic('stream', '=== EOG ===')
    
    elif mode == 'condense':
        if kwargs['target'] == 'session':
            page.pubsub.send_all_on_topic('condense-session-out', condense_output)
        if kwargs['target'] == 'span':
            page.pubsub.send_all_on_topic('condense-span-out', {'prompt': message, 'output': condense_output})

    else:
        await '=== EOG ==='

def load_model(page):
    global state

    if not state['model_path']:
        return
        
    grammar = None
    grammar_path = state['model_grammar']
    if grammar_path:
        try:
            grammar = LlamaGrammar.from_file(grammar_path)
        except:
            # TODO: snackbars eventually better triggered via pubsub message to dedicated function -> we wouldn't need to pass around page
            page.show_snack_bar(ft.SnackBar(ft.Text('Could not load LlamaGrammar: '+grammar_path), open=True))
    
    if not Path(state['model_path']).exists():
        page.show_snack_bar(ft.SnackBar(ft.Text('Could not find model file: '+state['model_path']), open=True))
        return
    
    return LLamaCppStream[str, str](
        'chat_stream',
        lambda prompt: state['sys_prompt']+'\n'+prompt,
        model=state['model_path'],
        n_ctx=state['model_ctx'],
        temperature=state['param_temp'],
        stop=[state['tag_user'], state['tag_asst']],
        top_k=state['param_topk'],
        top_p=state['param_topp'],
        repeat_penalty=state['param_reppen'],
        grammar=grammar
    )

schema_defaults = {
    'id': None,
    'at': None,
    'model_path': None,
    'model_grammar': None,
    'model_ctx': 4096,
    'sys_prompt': '### System:\nThis is a system prompt, please behave and help the user.',
    'tag_user': '### User:',
    'tag_asst': '### Assistant:',
    'transcript': '',
    'summary': '',
    'param_temp': 0.8,
    'param_topk': 40,
    'param_topp': 0.1,
    'param_reppen': 1.18,
    'condense_prompt': 'Summarize the text above.',
    'parent': None,
    'text_size': 14,
    'theme': 'auto',
    'tag_user_end': '',
    'tag_asst_end': ''
}

schema_sqlite = 'state (%s INTEGER PRIMARY KEY, %s TIMESTAMP DEFAULT CURRENT_TIMESTAMP, %s TEXT, %s TEXT, %s INTEGER, %s TEXT, %s TEXT, %s TEXT, %s TEXT, %s TEXT, %s REAL, %s INTEGER, %s REAL, %s REAL, %s TEXT, %s INTEGER, %s INTEGER, %s TEXT, %s TEXT, %s TEXT);' % tuple(k for k in schema_defaults.keys())


# flet app

def main(page: ft.Page):

    # load state
    global state
    dotdir = Path.home() / '.spica'
    dotdir.mkdir(exist_ok=True)
    dbfile = dotdir / 'state.db'
    con = sqlite3.connect(dbfile, check_same_thread=False)
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS '+schema_sqlite)
    con.commit()
    res = cur.execute('SELECT * FROM state ORDER BY at DESC')
    last_state = res.fetchone()
    if last_state:
        state = {label: param if param != 'NULL' else None for label, param in zip(schema_defaults.keys(), last_state)}
    else:
        state = schema_defaults

    # load model
    chat_stream = load_model(page)

    # page setup
    page.title = 'spica'
    page.window_resizable = True
    page.window_maximizable = False
    page.window_frameless = False
    page.window_title_bar_hidden = True
    page.window_min_height = 600
    page.window_min_width = 800
    page.scroll = ft.ScrollMode.ALWAYS
    page.auto_scroll = True
    routes = [
        ('/chat', 'Chat'),
        ('/sessions', 'Sessions'),
        ('/settings', 'Settings'),
        ('/chat-edit', 'Edit chat')
    ]
    pr = ft.Ref[ft.ProgressRing]()
    rail = ft.NavigationRail(
        selected_index=1,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=400,
        leading=ft.ProgressRing(ref=pr, width=64, height=64, stroke_width=10, value=len(state['transcript'])/state['model_ctx']),
        group_alignment=-0.9,
        expand=True,
        destinations=[
            ft.NavigationRailDestination(
                icon_content=ft.Icon(name=ft.icons.SHORT_TEXT, size=30),
                selected_icon_content=ft.Icon(name=ft.icons.SHORT_TEXT, size=30),
                label=routes[0][1],  # chat
                padding=ft.padding.only(top=24)
            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(name=ft.icons.GRID_VIEW_ROUNDED, size=22),
                selected_icon_content=ft.Icon(name=ft.icons.GRID_VIEW_ROUNDED, size=22),
                label=routes[1][1]  # sessions
            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(name=ft.icons.SETTINGS_ROUNDED, size=22),
                selected_icon_content=ft.Icon(name=ft.icons.SETTINGS_ROUNDED, size=22),
                label=routes[2][1]  # settings
            )
        ],
        on_change=lambda e: page.go(routes[e.control.selected_index][0])
    )
    chat = ft.Ref[ft.Column]()
    new_message = ft.Ref[ft.TextField]()
    send_button = ft.Ref[ft.ElevatedButton]()
    cancel_button = ft.Ref[ft.ElevatedButton]()
    sessions_grid = ft.Ref[ft.GridView]()
    optional_input1 = ft.Ref[ft.TextField]()
    optional_input2 = ft.Ref[ft.TextField]()
    
    # load theme
    def set_theme(value):
        if value == 'auto':
            return ft.ThemeMode.SYSTEM
        elif value == 'light':
            return ft.ThemeMode.LIGHT
        elif value == 'dark':
            return ft.ThemeMode.DARK

    def get_bgcolor():
        # determine current bgcolor
        if page.theme_mode == ft.ThemeMode.DARK:
            bgcolor = page.dark_theme.color_scheme.surface
        elif page.theme_mode == ft.ThemeMode.LIGHT:
            bgcolor = page.theme.color_scheme.surface
        elif page.theme_mode == ft.ThemeMode.SYSTEM:
            if page.platform_brightness == ft.ThemeMode.DARK:
                bgcolor = page.dark_theme.color_scheme.surface
            else:
                bgcolor = page.theme.color_scheme.surface
        return bgcolor

    def listener_brightness(e):
        route_change(ft.RouteChangeEvent(route=page.route))  # reload current view

    page.theme_mode = set_theme(state['theme'])
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            on_secondary_container='#1D1B1F',
            on_secondary='#D1C7BB',
            on_surface='#1D1B1F',
            on_surface_variant='#1D1B1F',
            on_inverse_surface='#1D1B1F',
            primary='#1D1B1F',
            on_primary='#D1C7BB',
            on_tertiary='#D1C7BB',
            on_primary_container='#1D1B1F',
            on_tertiary_container='#1D1B1F',
            on_error='#1D1B1F',
            on_error_container='#1D1B1F',
            on_background='#1D1B1F',
            secondary='#1D1B1F',
            tertiary='#1D1B1F',
            primary_container='#D1C7BB',
            secondary_container='#D1C7BB',
            tertiary_container='#D1C7BB',
            error='#D1C7BB',
            error_container='#D1C7BB',
            surface_tint=None,
            shadow='#D1C7BB',
            scrim='#1D1B1F',
            background='#D1C7BB',
            surface='#D1C7BB',
            surface_variant='#D1C7BB',
            outline='#D1C7BB',
            outline_variant='#D1C7BB',
            inverse_primary='#D1C7BB',
            inverse_surface='#FFFBFF'
        ),
        color_scheme_seed='#D1C7BB',
    )
    page.dark_theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary='#D1C7BB',
            surface=None,
            surface_tint='#8B8B94',
            shadow='#6B6564',
            secondary='#6B6564',
            on_secondary='#D1C7BB',
            secondary_container='#1D1B1F'
        ),
        color_scheme_seed='#8C9498',
    )
    page.on_platform_brightness_change = listener_brightness
    for platform in ['android', 'ios', 'macos', 'linux', 'windows']:
        setattr(page.theme.page_transitions, platform, ft.PageTransitionTheme.NONE)
        setattr(page.dark_theme.page_transitions, platform, ft.PageTransitionTheme.NONE)

    # events
    def on_message(topic, message: Message):
        chat.current.controls.append(
            ft.ResponsiveRow(
                [
                    ft.Icon(name=ft.icons.PERSON_PIN, color=ft.colors.SECONDARY, size=22, col={"md": 1}),
                    ft.Text(
                        spans=[ft.TextSpan(
                            text=message.text,
                            on_click=action_chat,
                            data={'action': 'edit_display'}
                        )],
                        color=ft.colors.INVERSE_SURFACE,
                        bgcolor=ft.colors.SECONDARY,
                        selectable=True,
                        size=state['text_size'],
                        col={"md": 10}
                    ),
                    ft.SafeArea(content=ft.VerticalDivider(width=8), col={"md": 1})
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            )
        )

        sanity = True
        if not state['model_path']:
            page.show_snack_bar(ft.SnackBar(ft.Text('Please select a model file.'), open=True))
            sanity = False
        else:
            if not state['tag_user'] or not state['tag_asst']:
                page.show_snack_bar(ft.SnackBar(ft.Text('Please define user and assistant tag.'), open=True))
                sanity = False
        if not pr.current.value and len(state['transcript']) > 0:  # llm is already running
            page.show_snack_bar(ft.SnackBar(ft.Text('Model is generating. Please wait..'), open=True))
            sanity = False
        if not sanity:            
            chat.current.controls[-1].controls[1].spans[0].text += ' [NOT SENT]'
            chat.current.controls[-1].controls[1].color = ft.colors.RED_100
            chat.current.controls[-1].controls[1].bgcolor = ft.colors.RED_900
            page.update()
            return

        chat.current.controls.append(
            ft.ResponsiveRow(
                [
                    ft.Icon(name=ft.icons.ASSISTANT, color=ft.colors.SECONDARY, size=22, col={"md": 1}),
                    ft.Text('', color=ft.colors.ON_SURFACE, selectable=True, size=state['text_size'], col={"md": 10}),
                    ft.SafeArea(content=ft.VerticalDivider(width=8), col={"md": 1})
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            )
        )
        save_state()
        chat_stream.running = True
        pr.current.value = None
        new_message.current.suffix = cancel_button.current
        new_message.current.focus()
        page.update()

        # call to llm
        state['transcript'] += '\n'+state['tag_user']+' '+message.text+state['tag_user_end']
        state['transcript'] += '\n'+state['tag_asst']+' '
        asyncio.run(stream_output(page, chat_stream, state['transcript']))

    def action_chat(e):
        data = e.control.data
        delim = '```````````````'
        if page.dialog:
            page.dialog.open = False

        if data['action'] == 'send':
            page.pubsub.send_all_on_topic('input', Message(user=page.session.get('user_name'), text=new_message.current.value))
            new_message.current.value = ''

        elif data['action'] == 'cancel':
            chat_stream.running = False
            pr.current.value = len(state['transcript']) / state['model_ctx']
            new_message.current.suffix = send_button.current

        elif data['action'] == 'edit_display':
            page.go(routes[3][0])
        
        elif data['action'] == 'edit_condense':
            checkbox_values = [row.controls[-1].value for row in chat.current.controls]
            ts = state['transcript'].strip().replace(state['tag_user'], delim+state['tag_user']).replace(state['tag_asst'], delim+state['tag_asst'])
            ts_split = list(filter(None, ts.split(delim)))
            assert len(ts_split) == len(checkbox_values)
            span = ''.join([msg for msg, checked in zip(ts_split, checkbox_values) if checked])
            page.pubsub.send_all_on_topic('condense-span-in', span)
            pr.current.value = None

        elif data['action'] == 'edit_delete':
            checkbox_values = [row.controls[-1].value for row in chat.current.controls]
            ts = state['transcript'].strip().replace(state['tag_user'], delim+state['tag_user']).replace(state['tag_asst'], delim+state['tag_asst'])
            ts_split = list(filter(None, ts.split(delim)))
            assert len(ts_split) == len(checkbox_values)
            state['transcript'] = ''.join([msg for msg, checked in zip(ts_split, checkbox_values) if not checked])
            save_state()
            pr.current.value = len(state['transcript']) / state['model_ctx']
            route_change(ft.RouteChangeEvent(route=page.route))  # reload current view to reflect changes
        
        elif data['action'] == 'edit_selectall':
            checkbox_values = [row.controls[-1].value for row in chat.current.controls]
            if len(set(checkbox_values)) == 1:
                if checkbox_values[0] == True:
                    new_value = False
                else:
                    new_value = True
            else:
                new_value = True
            for row in chat.current.controls:
                row.controls[-1].value = new_value
        
        page.update()

    def action_session(e):
        global state
        data = e.control.data

        if page.dialog:
            page.dialog.open = False
        
        if not pr.current.value and data['action'] != 'delete_session':
            page.show_snack_bar(ft.SnackBar(ft.Text('Model is generating. Please wait..'), open=True))
            return

        if data['action'] == 'load_session':
            reload = data.get('reload', None)
            if reload:
                load_state(data['target_id'], reload=True)
            else:
                load_state(data['target_id'])
        
        elif data['action'] == 'new_session':
            state = schema_defaults
            
            # auto-populate with last known model_path, user/asst tags
            res = cur.execute('SELECT * FROM state ORDER BY at DESC')
            last_state = res.fetchone()
            if last_state:
                state['model_path'] = last_state[2]
                state['tag_user'] = last_state[6]
                state['tag_asst'] = last_state[7]
                state['text_size'] = last_state[16]
                state['theme'] = last_state[17]

            save_state(insert=True)

        elif data['action'] in ['sum_session', 'sum_regenerate']:
            load_state(data['target_id'])
            if state['summary'] == '' or data['action'] == 'sum_regenerate':
                page.pubsub.send_all_on_topic('condense-session-in', state['transcript'])
                pr.current.value = None  # llm running
                if data['action'] == 'sum_regenerate':
                    page.dialog.open = False
                page.show_snack_bar(ft.SnackBar(ft.Text('Summarizing session'), open=True))
                page.update()
                return
            else:
                page.dialog = ft.AlertDialog(
                    modal=False,
                    title=ft.Text('Session summary'),
                    content=ft.Text(state['summary']),
                    actions=[
                        ft.TextButton('Regenerate', on_click=action_session, data={'action': 'sum_regenerate', 'target_id': data['target_id']}),
                        ft.TextButton('Close', on_click=action_session, data={'action': 'close_summary'})
                    ],
                    actions_alignment=ft.MainAxisAlignment.END
                )
                page.dialog.open = True
                page.update()
                return
        
        elif data['action'] == 'duplicate_session':
            load_state(data['target_id'])
            state['parent'] = data['target_id']
            save_state(insert=True)
        
        elif data['action'] == 'delete_session':
            cur.execute('DELETE FROM state WHERE rowid = %s' % (data['target_id']))
            con.commit()

            if data['target_id'] == state['id']:
                res = cur.execute('SELECT * FROM state ORDER BY at DESC')
                last_state = res.fetchone()
                if last_state:
                    state = {label: param if param != 'NULL' else None for label, param in zip(schema_defaults.keys(), last_state)}
                    pr.current.value = len(state['transcript']) / state['model_ctx']  # llm context usage
                else:
                    state = schema_defaults
                    save_state(insert=True)
            
            page.dialog.open = False
                
        tv = data.get('target_view', None)
        if not tv or tv == page.route:
            route_change(ft.RouteChangeEvent(route=page.route))  # reload current view
        else:  # mark the target navrail destination + go
            if tv == routes[0][0]:
                rail.selected_index = 0
            elif tv == routes[1][0]:
                rail.selected_index = 1
            elif tv == routes[2][0]:
                rail.selected_index = 2
            page.go(tv)

    def action_setting(e):

        if hasattr(e, 'files'):
            if e.files:
                if hasattr(e.files[0], 'path'):
                    if e.control.file_type == ft.FilePickerFileType.CUSTOM:
                        state['model_path'] = str(Path(e.files[0].path))
                    else:
                        state['model_grammar'] = str(Path(e.files[0].path))
                    
                    global chat_stream  # reload model
                    chat_stream = load_model(page)
                    route_change(ft.RouteChangeEvent(route=page.route))  # reload current view

        if hasattr(e.control, 'data'):
            if e.control.data:
                target = e.control.data.get('target', None)
                value = e.control.value
                if isinstance(value, str):
                    if value.isnumeric() and '.' in value:
                        value = float(value)
                    elif value.isnumeric() and '.' not in value:
                        value = int(value)
                if target:
                    state[target] = value
                    page.update()
                if target == 'theme':
                    page.theme_mode = set_theme(value)
                    route_change(ft.RouteChangeEvent(route=page.route))  # reload current view

    def on_keyboard(e: ft.KeyboardEvent):
        
        # chat: ctrl/cmd + enter to send message
        if page.platform in ['win', 'linux'] and e.ctrl == True and e.key == 'Enter' or e.meta == True and e.key == 'Enter':
            page.pubsub.send_all_on_topic('input', Message(user=page.session.get('user_name'), text=new_message.current.value))
            new_message.current.value = ''
            page.update()

        # chat: ctrl/cmd + plus-sign to increase font size for messages
        if e.ctrl == True and e.key == '+' or e.meta == True and e.key == '+':
            for row in chat.current.controls:
                row.controls[1].size += 2
            state['text_size'] += 2
            page.update()
        
        # chat: ctrl/cmd + minus-sign to decrease font size for messages
        if e.ctrl == True and e.key == '-' or e.meta == True and e.key == '-':
            for row in chat.current.controls:
                row.controls[1].size -= 2
            state['text_size'] -= 2
            page.update()

        if page.route == routes[3][0]:

            # chat edit mode: ctrl/cmd + a to de/select all messages
            if e.ctrl == True and e.key == 'a' or e.meta == True and e.key == 'a':
                checkbox_values = [row.controls[-1].value for row in chat.current.controls]
                if len(set(checkbox_values)) == 1:
                    if checkbox_values[0] == True:
                        new_value = False
                    else:
                        new_value = True
                else:
                    new_value = True
                for row in chat.current.controls:
                    row.controls[-1].value = new_value
            
            # chat edit mode: exit
            if e.key == 'Escape':
                page.go(routes[0][0])

    def dialog_del_sess(e):
        data = e.control.data
        tv = data.get('target_view', None)

        page.dialog = ft.AlertDialog(
            modal=False,
            title=ft.Text(data['message']),
            actions=[
                ft.TextButton('Yes', on_click=action_session, data={'action': 'delete_session', 'target_id': data['target_id'], 'target_view': tv}),
                ft.TextButton('No', on_click=action_session, data={'action': 'delete_session', 'target_id': data['target_id'], 'target_view': tv})
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )
        page.dialog.open = True
        page.update()

    def dialog_del_msg(e):
        data = e.control.data

        page.dialog = ft.AlertDialog(
            modal=False,
            title=ft.Text(data['message']),
            actions=[
                ft.TextButton('Yes', on_click=action_chat, data={'action': 'edit_delete'}),
                ft.TextButton('No', on_click=action_chat, data={'action': 'edit_delete'})
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )
        page.dialog.open = True
        page.update()

    def listener_stream(topic, message: StreamOutput):
        if message == '=== EOG ===':  # not message.data
            save_state()
            pr.current.value = len(state['transcript']) / state['model_ctx']
            new_message.current.suffix = send_button.current
            route_change(ft.RouteChangeEvent(route=page.route))  # reload current view (and save state)
            if pr.current.value >= 0.9:
                page.show_snack_bar(ft.SnackBar(ft.Text('Reduce context window usage by condensing passages. 👉 Select Text'), open=True))
        else:
            try:
                chat.current.controls[-1].controls[1].value += message.data
            except:
                pass  # we're not in chat view
            pr.current.value = None  # llm running
            state['transcript'] += message.data
        
        page.update()

    def listener_condense(topic, message):
        if topic.endswith('-out'):
            if topic == 'condense-session-out':
                state['summary'] = message

            if topic == 'condense-span-out':                
                span_orig = message['prompt'].split('======== TEXT BEGIN ========\n\n')[-1].split('\n\n======== TEXT END ========')[0]
                span_new = state['tag_asst']+' Summary: '+message['output']
                state['transcript'] = state['transcript'].replace(span_orig, span_new)
            
            save_state()
            pr.current.value = len(state['transcript']) / state['model_ctx']
            route_change(ft.RouteChangeEvent(route=page.route))  # reload current view
            page.show_snack_bar(ft.SnackBar(ft.Text('Condensation finished.'), open=True))
        
        if topic.endswith('-in'):
            if not pr.current.value:  # llm is already running
                page.show_snack_bar(ft.SnackBar(ft.Text('Model is generating. Please wait..'), open=True))
                return
            
            condense_text = '======== TEXT BEGIN ========\n\n'+message+'\n\n======== TEXT END ========\n\n'+state['tag_user']+' '
            condense_prompt = state['condense_prompt']
            if topic == 'condense-session-in':
                target = 'session'
            if topic == 'condense-span-in':
                target = 'span'
                page.go(routes[0][0])  # exit edit mode
            combined_prompt = condense_text + condense_prompt + '\n'+state['tag_asst']+' '

            try:
                condense_loop = asyncio.get_running_loop()
            except:
                condense_loop = asyncio.new_event_loop()
            condense_loop.run_until_complete(stream_output(page, chat_stream, combined_prompt, mode='condense', target=target))
            condense_loop.close()

    def load_state(target, reload=False):
        global state
        if not pr.current.value and len(state['transcript']) > 0:
            page.show_snack_bar(ft.SnackBar(ft.Text('Model is generating. The process can be canceled in chat view.'), open=True))
            return
        res = cur.execute('SELECT * FROM state WHERE rowid = %s' % (target))
        previous = {label: param for label, param in zip(schema_defaults.keys(), res.fetchone())}
        state = previous
        pr.current.value = len(state['transcript']) / state['model_ctx']  # llm context usage

        if reload:
            global chat_stream
            chat_stream = load_model(page)

    def save_state(summary=False, insert=False):
        global state

        # generate summary
        if summary:
            page.pubsub.send_all_on_topic('condense-session-in', state['transcript'])
            
        # write to table
        s = { k: ('NULL' if v is None else str(v)) for k, v in state.items() }
        row_id = s['id']
        del s['id']
        s['transcript'] = s['transcript'].replace('"', '\'')
        s['summary'] = s['summary'].replace('"', '\'')

        if insert:
            del s['at']
            columns = ', '.join(s.keys())
            values = '"'+'", "'.join(s.values())+'"'
            cur.execute('INSERT INTO state (%s) VALUES (%s)' % (columns, values))
            # update metadata in state variable
            res = cur.execute('SELECT * FROM state ORDER BY at DESC')
            last_state = res.fetchone()
            state['id'] = last_state[0]
            state['at'] = last_state[1]
        else:
            s['at'] = str(datetime.now())
            columns = ', '.join(s.keys())
            values = '"'+'", "'.join(s.values())+'"'
            cur.execute('UPDATE state SET (%s) = (%s) WHERE rowid = %s' % (columns, values, row_id))
                    
        con.commit()

    if not last_state:
        save_state(insert=True)

    # more page setup
    pick_model_dlg = ft.FilePicker(on_result=action_setting)
    pick_grammar_dlg = ft.FilePicker(on_result=action_setting)
    ft.ElevatedButton(
        ref=send_button,
        text='Send',
        tooltip='Send',
        on_click=action_chat,
        data={'action': 'send'}
    )
    ft.ElevatedButton(
        ref=cancel_button,
        text='Cancel',
        tooltip='Cancel',
        color=ft.colors.RED_100,
        bgcolor=ft.colors.RED_900,
        on_click=action_chat,
        data={'action': 'cancel'}
    )

    page.on_keyboard_event = on_keyboard
    page.pubsub.subscribe_topic('input', on_message)
    page.pubsub.subscribe_topic('stream', listener_stream)
    page.pubsub.subscribe_topic('condense-session-in', listener_condense)
    page.pubsub.subscribe_topic('condense-span-in', listener_condense)
    page.pubsub.subscribe_topic('condense-session-out', listener_condense)
    page.pubsub.subscribe_topic('condense-span-out', listener_condense)
    
    # route management
    def route_change(e: ft.RouteChangeEvent):
        global state
        page.views.clear()
        page.overlay.clear()
        chat_edit = (page.route == routes[3][0])
        bgcolor = get_bgcolor()  # bgcolor for views

        if page.route == routes[0][0] or chat_edit:  # chat + chat-edit

            if chat_edit:  # edit mode: display other actions in place of the send button
                input_row = ft.Row(
                    [
                        ft.ElevatedButton(
                            icon=ft.icons.ARROW_BACK,
                            text='Back',
                            tooltip='Back',
                            on_click=lambda e: page.go(routes[0][0])
                        ),
                        ft.VerticalDivider(width=8),
                        ft.ElevatedButton(
                            text='Select all',
                            tooltip='Select all',
                            on_click=action_chat,
                            data={'action': 'edit_selectall'}
                        ),
                        ft.VerticalDivider(width=8),
                        ft.ElevatedButton(
                            text='Condense',
                            tooltip='Condense',
                            color=ft.colors.RED_100,
                            bgcolor=ft.colors.RED_900,
                            on_click=action_chat,
                            data={'action': 'edit_condense'}
                        ),
                        ft.VerticalDivider(width=8),
                        ft.ElevatedButton(
                            text='Delete',
                            tooltip='Delete',
                            color=ft.colors.RED_100,
                            bgcolor=ft.colors.RED_900,
                            on_click=dialog_del_msg,
                            data={'message': 'Are you sure?'}
                        )
                    ]
                )
            else:
                if not pr.current.value and state['model_path']:
                    suffix = cancel_button.current
                else:
                    suffix = send_button.current
                input_row = ft.TextField(
                    ref=new_message,
                    multiline=True,
                    autofocus=True,
                    expand=True,
                    border_color=ft.colors.TRANSPARENT,
                    focused_border_color=ft.colors.TRANSPARENT, 
                    suffix=suffix
                )

            page.views.append(
                ft.View(
                    routes[0][0],
                    [
                        ft.Row(
                            [
                                ft.Column(
                                    [
                                        ft.Divider(height=24),
                                        rail,
                                    ],
                                    expand=0.5
                                ),
                                ft.VerticalDivider(width=1, opacity=0.0),
                                ft.Column(
                                    [
                                        ft.Container(
                                            content=ft.GestureDetector(
                                                on_tap=action_chat,
                                                data={'action': 'edit_display'},
                                                content=ft.Column(ref=chat, scroll=ft.ScrollMode.ALWAYS, auto_scroll=True, expand=True),
                                                expand=True
                                            )
                                        ),
                                        ft.Divider(height=1, opacity=0.0),
                                        ft.Container(
                                            content=ft.Row(
                                                [
                                                    input_row,
                                                    ft.VerticalDivider(width=8)
                                                ],
                                                expand=True
                                            ),
                                        ),
                                        ft.Divider(height=1, opacity=0.0)
                                    ],
                                    expand=True,
                                    alignment=ft.MainAxisAlignment.END,
                                    scroll=ft.ScrollMode.ALWAYS,
                                    auto_scroll=True,
                                )
                            ],
                            expand=True
                        )
                    ],
                    bgcolor=bgcolor
                )
            )

            if 'state' in globals():
                ts = state.get('transcript', '')
            else:
                ts = ''

            if ts != '':
                delim = '```````````````'
                ts = ts.replace(state['tag_user'], delim+state['tag_user'])
                ts = ts.replace(state['tag_asst'], delim+state['tag_asst'])

                for msg in ts.split(delim):

                    if chat_edit:  # edit mode: display checkboxes for each message
                        trailer = ft.Checkbox(value=False, col={"md": 1})
                    else:
                        trailer = ft.SafeArea(content=ft.VerticalDivider(width=8), col={"md": 1})
                    
                    if msg.startswith(state['tag_user']):
                        text = msg.replace(state['tag_user'], '').strip()
                        if len(state['tag_user_end']) > 3:
                            text = text.replace(state['tag_user_end'], '')
                        chat.current.controls.append(
                            ft.ResponsiveRow(
                                [
                                    ft.Icon(name=ft.icons.PERSON_PIN, color=ft.colors.SECONDARY, size=22, col={"md": 1}),
                                    ft.Text(
                                        spans=[ft.TextSpan(
                                            text=text,
                                            on_click=action_chat,
                                            data={'action': 'edit_display'}
                                        )],
                                        color=ft.colors.INVERSE_SURFACE,
                                        bgcolor=ft.colors.SECONDARY,
                                        selectable=True,
                                        size=state['text_size'],
                                        col={"md": 10},
                                    ),
                                    trailer
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            )
                        )
                    elif msg.startswith(state['tag_asst']):
                        text = msg.replace(state['tag_asst'], '').strip()
                        if len(state['tag_asst_end']) > 3:
                            text = text.replace(state['tag_asst_end'], '')
                        chat.current.controls.append(
                            ft.ResponsiveRow(
                                [
                                    ft.Icon(name=ft.icons.ASSISTANT, color=ft.colors.SECONDARY, size=22, col={"md": 1}),
                                    ft.Text(
                                        spans=[ft.TextSpan(
                                            text=text,
                                            on_click=action_chat,
                                            data={'action': 'edit_display'}
                                        )],
                                        color=ft.colors.ON_SURFACE,
                                        selectable=True,
                                        size=state['text_size'],
                                        col={"md": 10},
                                    ),
                                    trailer
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            )
                        )
                    
        if page.route == routes[1][0]:  # sessions

            page.views.append(
                ft.View(
                    routes[1][0],
                    [
                        ft.Row(
                            [
                                ft.Column(
                                    [
                                        ft.Divider(height=24),
                                        rail,
                                    ],
                                    expand=0.5
                                ),
                                ft.VerticalDivider(width=1, opacity=0.0),
                                ft.GridView(
                                    ref=sessions_grid,
                                    expand=1,
                                    runs_count=2,
                                    max_extent=500,
                                    child_aspect_ratio=2.5,
                                    spacing=5,
                                    run_spacing=5,
                                )
                            ],
                            expand=True
                        )
                    ],
                    bgcolor=bgcolor
                )
            )
            
            res = cur.execute('SELECT * FROM state ORDER BY at DESC')
            sessions = res.fetchall()
            sessions = [()]+sessions

            for i, sess in enumerate(sessions):

                if i == 0:  # new session
                    tile = ft.ListTile(
                        leading=ft.Icon(ft.icons.ADD, size=40),
                        title=ft.Text(spans=[ft.TextSpan('\nnew\n', ft.TextStyle(weight=ft.FontWeight.BOLD))]),
                        subtitle=ft.Text('New session'),
                        on_click=action_session,
                        data={'action': 'new_session', 'target_view': routes[0][0]}
                    )
                else:
                    popupmenu = ft.PopupMenuButton(
                        icon=ft.icons.MORE_VERT,
                        items=[
                            ft.PopupMenuItem(
                                text="Load Model",
                                on_click=action_session,
                                data={'action': 'load_session', 'target_id': sess[0], 'reload': True}
                            ),
                            ft.PopupMenuItem(
                                text="Summary",
                                on_click=action_session,
                                data={'action': 'sum_session', 'target_id': sess[0]}
                            ),
                            ft.PopupMenuItem(
                                text="Duplicate",
                                on_click=action_session,
                                data={'action': 'duplicate_session', 'target_id': sess[0]}
                            ),
                            ft.PopupMenuItem(
                                text="Delete",
                                on_click=dialog_del_sess,
                                data={'message': 'Are you sure?', 'target_id': sess[0]}
                            ),
                        ],
                    )

                    # meta
                    timestamp = datetime.strptime(sess[1].split('.')[0], '%Y-%m-%d %H:%M:%S')
                    date, time = datetime.strftime(timestamp, '%m/%d %H:%M').split(' ')
                    meta = ft.Text(
                        spans=[
                            ft.TextSpan('\n '+date, ft.TextStyle(weight=ft.FontWeight.BOLD)),
                            ft.TextSpan('  '+time+'\n')
                        ]
                    )

                    tile = ft.ListTile(
                        leading=ft.ProgressRing(width=24, height=24, stroke_width=4, value=len(sess[8])/sess[4]),
                        title=ft.Container(
                            content=meta,
                            padding=ft.padding.only(left=7)
                        ),
                        subtitle=ft.Container(
                            content=ft.Text(
                                sess[9],
                                text_align=ft.TextAlign.LEFT,
                                overflow=ft.TextOverflow.FADE,
                                max_lines=2
                            ),
                            padding=ft.padding.only(10, 0, -20, 0),
                        ),
                        content_padding=ft.padding.only(left=24, right=16),
                        on_click=action_session,
                        data={'action': 'load_session', 'target_id': sess[0], 'target_view': routes[0][0]},
                        trailing=popupmenu
                    )

                sessions_grid.current.controls.append(
                    ft.Card(
                        content=tile
                    )
                )
        
        if page.route == routes[2][0]:  # settings

            page.overlay.append(pick_model_dlg)
            page.overlay.append(pick_grammar_dlg)
            page.views.append(
                ft.View(
                    routes[2][0],
                    [
                        ft.Row(
                            [
                                ft.Column(
                                    [
                                        ft.Divider(height=24),
                                        rail,
                                    ],
                                    expand=0.5
                                ),
                                ft.VerticalDivider(width=1, opacity=0.0),
                                ft.GridView(
                                    expand=True,
                                    runs_count=2,
                                    max_extent=800,
                                    child_aspect_ratio=0.8,
                                    horizontal=False,
                                    spacing=5,
                                    run_spacing=5,
                                    controls=[
                                        ft.Card(  # model-specific parameter
                                            content=ft.ListTile(
                                                title=ft.Container(
                                                    content=ft.Text('Model'),
                                                    padding=ft.padding.only(7, 7, 7, 14)
                                                ),
                                                subtitle=ft.Container(
                                                    content=ft.Column(
                                                        [
                                                            ft.TextField(
                                                                label='Model File',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=(state['model_path'].split('/')[-1] if state['model_path'] else 'Select file'),
                                                                on_focus=lambda _: pick_model_dlg.pick_files(file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=['gguf']),
                                                                on_blur=lambda _: save_state(),
                                                                suffix_icon=ft.icons.SEARCH_ROUNDED,
                                                                disabled=(len(state['transcript']) > 0)
                                                            ),
                                                            ft.TextField(
                                                                label='Context Size',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['model_ctx'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'model_ctx'},
                                                                disabled=(len(state['transcript']) > 0)
                                                            ),
                                                            ft.TextField(
                                                                label='Start User Tag',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['tag_user'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'tag_user'},
                                                                disabled=(len(state['transcript']) > 0)
                                                            ),
                                                            ft.TextField(
                                                                ref=optional_input1,
                                                                label='End User Tag',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['tag_user_end'] or 'Optional',
                                                                on_change=action_setting,
                                                                on_focus=lambda _: setattr(optional_input1.current, 'value', state['tag_user_end'] or '') or page.update(),
                                                                on_blur=lambda _: setattr(optional_input1.current, 'value', state['tag_user_end'] or 'Optional') or save_state() or page.update(),
                                                                data={'target': 'tag_user_end'},
                                                                disabled=(len(state['transcript']) > 0)
                                                            ),
                                                            ft.TextField(
                                                                label='Start Assistant Tag',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['tag_asst'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'tag_asst'},
                                                                disabled=(len(state['transcript']) > 0)
                                                            ),
                                                            ft.TextField(
                                                                ref=optional_input2,
                                                                label='End Assistant Tag',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['tag_asst_end'] or 'Optional',
                                                                on_change=action_setting,
                                                                on_focus=lambda _: setattr(optional_input2.current, 'value', state['tag_asst_end'] or '') or page.update(),
                                                                on_blur=lambda _: setattr(optional_input2.current, 'value', state['tag_asst_end'] or 'Optional') or save_state() or page.update(),
                                                                data={'target': 'tag_asst_end'},
                                                                disabled=(len(state['transcript']) > 0)
                                                            )
                                                        ]
                                                    ),
                                                    padding=ft.padding.only(10, 0, -20, 0),
                                                ),
                                                content_padding=ft.padding.only(left=24, right=48),
                                            )
                                        ),
                                        ft.Card(  # prompts
                                            content=ft.ListTile(
                                                title=ft.Container(
                                                    content=ft.Text('Prompt'),
                                                    padding=ft.padding.only(7, 7, 7, 14)
                                                ),
                                                subtitle=ft.Container(
                                                    content=ft.Column(
                                                        [
                                                            ft.TextField(
                                                                label='System Prompt',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                multiline=True,
                                                                value=state['sys_prompt'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'sys_prompt'},
                                                                disabled=(len(state['transcript']) > 0)
                                                            ),
                                                            ft.TextField(
                                                                label='Condensation Prompt',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                multiline=True,
                                                                value=state['condense_prompt'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'condense_prompt'}
                                                            )
                                                        ]
                                                    ),
                                                    padding=ft.padding.only(10, 0, -20, 0),
                                                ),
                                                content_padding=ft.padding.only(left=24, right=48),
                                            )
                                        ),
                                        ft.Card(  # generation-specific parameter
                                            content=ft.ListTile(
                                                title=ft.Container(
                                                    content=ft.Text('Generation'),
                                                    padding=ft.padding.only(7, 7, 7, 14)
                                                ),
                                                subtitle=ft.Container(
                                                    content=ft.Column(
                                                        [
                                                            ft.TextField(
                                                                label='Temperature',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['param_temp'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'param_temp'}
                                                            ),
                                                            ft.TextField(
                                                                label='Top K',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['param_topk'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'param_topk'}
                                                            ),
                                                            ft.TextField(
                                                                label='Top P',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['param_topp'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'param_topp'}
                                                            ),
                                                            ft.TextField(
                                                                label='Repetition Penalty',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['param_reppen'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'param_reppen'}
                                                            ),
                                                            ft.TextField(
                                                                label='Constraining Grammar',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=(state['model_grammar'].split('/')[-1] if state['model_grammar'] else 'Select file'),
                                                                on_focus=lambda _: pick_grammar_dlg.pick_files(file_type=ft.FilePickerFileType.ANY),
                                                                on_blur=lambda _: save_state(),
                                                                suffix_icon=ft.icons.SEARCH_ROUNDED
                                                            )
                                                        ]
                                                    ),
                                                    padding=ft.padding.only(10, 0, -20, 0),
                                                ),
                                                content_padding=ft.padding.only(left=24, right=48),
                                            )
                                        ),
                                        ft.Card(  # ui
                                            content=ft.ListTile(
                                                title=ft.Container(
                                                    content=ft.Text('UI'),
                                                    padding=ft.padding.only(7, 7, 7, 14)
                                                ),
                                                subtitle=ft.Container(
                                                    content=ft.Column(
                                                        [
                                                            ft.Dropdown(
                                                                label='Theme',
                                                                text_size=14,
                                                                value=state['theme'],
                                                                options=[
                                                                    ft.dropdown.Option('auto'),
                                                                    ft.dropdown.Option('light'),
                                                                    ft.dropdown.Option('dark'),
                                                                ],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'theme'}
                                                            ),
                                                            ft.TextField(
                                                                label='Text Size',
                                                                text_size=14,
                                                                color=ft.colors.ON_SURFACE,
                                                                value=state['text_size'],
                                                                on_change=action_setting,
                                                                on_blur=lambda _: save_state(),
                                                                data={'target': 'text_size'}
                                                            )
                                                        ]
                                                    ),
                                                    padding=ft.padding.only(10, 0, -20, 0),
                                                ),
                                                content_padding=ft.padding.only(left=24, right=48),
                                            )
                                        )
                                    ]
                                )
                            ],
                            expand=True
                        )
                    ],
                    bgcolor=bgcolor
                )
            )
        
        page.update()

    def view_pop(view):
        if len(page.views) > 1:
            page.views.pop()
            top_view = page.views[-1]
        else:
            top_view = page.views[0]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    page.go(routes[1][0])  # default view: sessions

ft.app(target=main)           
