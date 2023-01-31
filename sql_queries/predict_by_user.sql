SELECT created_at,app_id,user_id,session_type FROM onmo.game_sessions 
where user_id in ('f3b51e88-594b-4bb9-b732-3f28859f2011','40f10905-96ed-4808-824e-c9b45e373919') 
and date(created_at) >= '2022-01-01' and app_id not like 'embed-subwaysurfer';